import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.rope import RotaryEmbedding, apply_rotary_emb
from typing import Union, Tuple, Optional, Iterator


def create_casual_mask(seq_len, device):
    mask_tensor = torch.full((1, 1, seq_len, seq_len), -2.3819763e38).to(device)
    mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
    mask = torch.triu(mask_tensor, diagonal=1).to(device)
    return mask


class GemmaAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        attn_logit_softcapping,
        head_dim,
        attn_type,
        sliding_window_size,
    ):
        super().__init__()
        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            num_kv_heads * head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            num_kv_heads * head_dim,
            bias=False,
        )

        self.o_proj = nn.Linear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
        )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling_factor = head_dim**-0.5
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_group = self.num_heads // self.num_kv_heads
        self.attn_type = attn_type
        self.sliding_window_size = sliding_window_size
        self.attn_logit_softcapping = attn_logit_softcapping

    def forward(
        self, hidden_states, freqs_cis, mask, output_attn, train_full_attn=True
    ):
        batch_size, seq_len, dim = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # batch_size, seq_len, dim -> batch_size, seq_len, num_heads, dim
        q = q.view(batch_size, -1, self.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        # apply positional embedding.
        q = apply_rotary_emb(q, freqs_cis=freqs_cis)
        k = apply_rotary_emb(k, freqs_cis=freqs_cis)
        if self.num_kv_heads != self.num_heads:
            # GQA
            # [batch_size, seq_len, num_kv_heads, head_dim]
            k = torch.repeat_interleave(k, self.num_queries_per_group, dim=2)
            v = torch.repeat_interleave(v, self.num_queries_per_group, dim=2)
        # perform QK^T
        # We want to get output like (batch_size,num_heads,seq_len,seq_len)
        # Thus, we must reshape the q and k
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        qk = torch.einsum("BHSD,BHFD->BHSF", q, k)
        qk.mul_(self.scaling_factor)
        # construct causal or sliding window mask to prevent
        # query to attend to keys outside the boundary
        if self.attn_type == "local":
            all_ones = torch.ones_like(mask)
            sliding_mask = torch.triu(
                all_ones, 1 - self.sliding_window_size
            ) * torch.tril(all_ones, self.sliding_window_size - 1)
            mask = torch.where(sliding_mask == 1, mask, -2.3819763e38)
        # logits ← soft_cap ∗ tanh(logits/soft_cap)
        if self.attn_logit_softcapping is not None:
            qk = qk / self.attn_logit_softcapping
            qk = torch.tanh(qk)
            qk = qk * self.attn_logit_softcapping

        scores = qk + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        # [batch_size, num_heads, seq_len, head_dim]
        attn_output = torch.einsum("BHSF,BHFD->BHSD", scores, v)
        output = attn_output.transpose(1, 2).contiguous()
        output = self.o_proj(output.reshape(batch_size, seq_len, -1))
        if output_attn:
            outputs = {
                "hidden_states": output,
                "attentions": scores,
                "query": q,
                "key": k,
                "value": v,
            }
            if train_full_attn:
                outputs["attentions"] = attn_output
            return outputs
        return {"hidden_states": output}


class Gemma2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        # gemma use approximate gelu
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(hidden_states)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        # using rsqrt (reciprocal square root) is equivalent to 1/RMS(X)
        # but directly perform division is less efficient than multiplication
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config, attention_type):
        super().__init__()
        self.attention = GemmaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            attn_logit_softcapping=config.attn_logit_softcapping,
            head_dim=config.head_dim,
            attn_type=attention_type,
            sliding_window_size=config.sliding_window_size,
        )
        self.mlp = Gemma2MLP(config=config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
        output_attn: Optional[bool],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        outputs = self.attention(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            mask=mask,
            output_attn=output_attn,
        )
        hidden_states = outputs["hidden_states"]
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        if output_attn:
            return {
                "hidden_states": hidden_states,
                "attentions": outputs["attentions"],
                "query": outputs["query"],
                "key": outputs["key"],
                "value": outputs["value"],
            }
        return {"hidden_states": hidden_states}


class GemmaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        num_hidden_layers = config.num_hidden_layers
        self.decoder_layers = nn.ModuleList(
            [
                GemmaDecoderLayer(
                    config=config, attention_type=config.attention_type[i]
                )
                for i in range(num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
        output_attn: Optional[bool],
    ) -> torch.Tensor:
        all_outputs = {
            "all_attn": [],
            "all_query": [],
            "all_key": [],
            "all_hs": [],
            "all_value": [],
        }

        for layer in self.decoder_layers:
            outputs = layer(
                hidden_states=hidden_states,
                freqs_cis=freqs_cis,
                mask=mask,
                output_attn=output_attn,
            )
            hidden_states = outputs["hidden_states"]
            if output_attn:
                all_outputs["all_attn"].append(outputs["attentions"])
                all_outputs["all_query"].append(outputs["query"])
                all_outputs["all_key"].append(outputs["key"])
                all_outputs["all_hs"].append(outputs["hidden_states"])
                all_outputs["all_value"].append(outputs["value"])

        hidden_states = self.norm(hidden_states)
        return {"hidden_states": hidden_states, "all_outputs": all_outputs}


class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_seq_len = config.max_position_embeddings
        self.hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        padding_idx = config.padding_idx
        num_attention_heads = config.num_attention_heads
        assert self.hidden_size % num_attention_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_attention_heads ({num_attention_heads})."
        )
        self.embedding_layer = nn.Embedding(vocab_size, self.hidden_size, padding_idx)
        self.model = GemmaModel(config=config)
        self.lm_head = nn.Linear(self.hidden_size, vocab_size, bias=False)
        self.rope_embeddings = RotaryEmbedding(
            theta=config.rope_theta,
            head_dim=config.head_dim,
            max_seq_len=self.max_seq_len,
        )

    def forward(
        self,
        input_ids,
        tok_idx=None,
        mask=None,
        num_logits_to_keep=0,
        output_attn=False,
    ):
        seq_len = input_ids.shape[1]
        freqs_cis = self.rope_embeddings(seq_len=seq_len, tok_idx=tok_idx)
        embeddings = self.embedding_layer(input_ids)
        # Gemma normalizes the embedding by sqrt(hidden_size).
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.hidden_size**0.5, dtype=embeddings.dtype)
        hidden_states = embeddings * normalizer
        if mask is None:
            mask = create_casual_mask(seq_len, device=input_ids.device)
        outputs = self.model(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            mask=mask,
            output_attn=output_attn,
        )
        hidden_states = outputs["hidden_states"]
        if output_attn:
            return outputs
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        return logits

    @torch.no_grad()
    def sampling(
        self,
        input_ids: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self(input_ids=input_ids)
        logits = logits[:, -1, ...]
        # greedy
        if temperatures is None:
            return torch.argmax(logits, dim=-1), logits

        # Apply temperature scaling.
        logits.div_(temperatures.unsqueeze(dim=1))

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Apply top-p, top-k.
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalization.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))

        next_token_ids = torch.multinomial(
            probs, num_samples=1, replacement=True
        ).squeeze(dim=-1)
        return next_token_ids, logits

    def generate(
        self,
        input_token_ids_tensor: torch.Tensor,
        output_len: int = 100,
        temperature: Union[float, None] = 0,
        top_p: float = 1.0,
        top_k: int = 100,
        eos_token_id: int = 107,
    ) -> Iterator[torch.Tensor]:
        batch_size = input_token_ids_tensor.shape[0]
        device = input_token_ids_tensor.device
        temperatures_tensor = (
            None
            if not temperature
            else torch.FloatTensor([temperature] * batch_size).to(device)
        )
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)

        for i in range(output_len):
            next_token_ids, _ = self.sampling(
                input_ids=input_token_ids_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
            )
            input_token_ids_tensor = torch.cat(
                [input_token_ids_tensor, next_token_ids.unsqueeze(1)], dim=1
            )
            if next_token_ids[0] == eos_token_id:
                break
            yield next_token_ids

    def load_weights(self, state_dict):
        key_mapping = {
            "model.embed_tokens": "embedding_layer",
            "layers": "decoder_layers",
            "self_attn": "attention",
        }
        keys_to_update = list(state_dict.keys())
        for old_key in keys_to_update:
            new_key = old_key
            for old_substring, new_substring in key_mapping.items():
                new_key = new_key.replace(old_substring, new_substring)
            if new_key != old_key:
                state_dict[new_key] = state_dict.pop(old_key)
        self.load_state_dict(state_dict, strict=False)
        del state_dict  # save memory.
        gc.collect()
        # tied embedding weight
        self.lm_head.weight = self.embedding_layer.weight
