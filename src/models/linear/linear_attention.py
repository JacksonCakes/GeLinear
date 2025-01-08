import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.models.rope import apply_rotary_emb
from src.models.linear.feature_map import HedgehogFeatureMap
from src.models.model import GemmaModel, create_casual_mask
from src.models.attentions import quadratic_attention, chunk_linear_attn


class GemmaLinearAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        train_attn=False,
        feature_dim=64,
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
        self.feature_map_q = HedgehogFeatureMap(
            num_heads=num_heads,
            head_dim=head_dim,
            feature_dim=feature_dim,
        )
        self.feature_map_k = HedgehogFeatureMap(
            num_heads=num_heads,
            head_dim=head_dim,
            feature_dim=feature_dim,
        )
        self.train_attn = train_attn
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling_factor = head_dim**-0.5
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_group = self.num_heads // self.num_kv_heads

    def forward(self, hidden_states, freqs_cis, mask):
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
        q = self.feature_map_q(q)
        k = self.feature_map_k(k)
        if self.train_attn:
            attn_output_pred, scores_pred = quadratic_attention(q=q, k=k, v=v)
            output = (
                attn_output_pred.transpose(1, 2)
                .contiguous()
                .view(batch_size, seq_len, -1)
            )
            output = self.o_proj(output)
            return {
                "hidden_states": output,
                "attn_scores_pred": scores_pred,
                "attn_output_pred": attn_output_pred,
            }

        output = chunk_linear_attn(q, k, v)
        output = (
            output.transpose(1, 2)
            .contiguous()[:, :seq_len, ...]
            .view(batch_size, seq_len, -1)
        )
        output = self.o_proj(output)
        return {"hidden_states": output}


class SubGemmaModel(nn.Module):
    def __init__(self, original_model: GemmaModel, rope_embeddings, layer_idx: list):
        super().__init__()
        self.embed_tokens = original_model.embed_tokens
        full_idx = list(range(layer_idx[0], layer_idx[-1] + 1))
        if layer_idx[0] % 2 == 0 or layer_idx[-1] % 2 == 0:
            raise ValueError(
                "Both the first and last indices in layer_idx must be odd."
            )
        self.layers = nn.ModuleList([original_model.layers[i] for i in full_idx])
        self.rope_embeddings = rope_embeddings
        for idx, layer in enumerate(self.layers):
            if idx % 2 == 0:
                q_proj = layer.self_attn.q_proj
                k_proj = layer.self_attn.k_proj
                v_proj = layer.self_attn.v_proj
                o_proj = layer.self_attn.o_proj

                layer.self_attn = GemmaLinearAttention(
                    hidden_size=layer.self_attn.hidden_size,
                    num_heads=layer.self_attn.num_heads,
                    num_kv_heads=layer.self_attn.num_kv_heads,
                    head_dim=layer.self_attn.head_dim,
                    train_attn=True,
                )
                layer.self_attn.q_proj = q_proj
                layer.self_attn.k_proj = k_proj
                layer.self_attn.v_proj = v_proj
                layer.self_attn.o_proj = o_proj
                layer.self_attn.feature_map_q.requires_grad = True
                layer.self_attn.feature_map_k.requires_grad = True

    def forward_layer(self, layer, hidden_states, freqs_cis, mask):
        """
        A wrapper that calls `layer(...)` and returns
        ONLY Tensor outputs so that we can checkpoint it.
        """
        outputs = layer(
            hidden_states=hidden_states,
            freqs_cis=freqs_cis,
            mask=mask,
        )
        hidden_states_out = outputs["hidden_states"]
        attn_scores_pred = outputs.get("attn_scores_pred", None)

        # checkpointing must return tensors in a tuple.
        return (hidden_states_out, attn_scores_pred)

    def forward(self, hidden_states: torch.Tensor, mask=None) -> torch.Tensor:
        all_outputs = {}
        seq_len = hidden_states.shape[1]
        freqs_cis = self.rope_embeddings(seq_len=seq_len, tok_idx=None)
        if mask is None:
            mask = create_casual_mask(seq_len, device=hidden_states.device)
        for idx, layer in enumerate(self.layers):
            if idx % 2 == 0:
                # ----------------------------------------
                # GRADIENT CHECKPOINTING
                # Instead of:
                #   outputs = layer(hidden_states, freqs_cis, mask)
                #
                # do:
                #   (hidden_states, attn_scores_pred) = checkpoint(...)
                # ----------------------------------------
                (hidden_states_out, attn_scores_pred) = checkpoint(
                    self.forward_layer,
                    layer,
                    hidden_states,
                    freqs_cis,
                    mask,
                    use_reentrant=False,
                )
                outputs = {
                    "hidden_states": hidden_states_out,
                }
                if attn_scores_pred is not None:
                    outputs["attn_scores_pred"] = attn_scores_pred

                hidden_states = hidden_states_out
            else:
                outputs = layer(
                    hidden_states=hidden_states,
                    freqs_cis=freqs_cis,
                    mask=mask,
                )

                hidden_states = outputs["hidden_states"]

            for k, v in outputs.items():
                all_outputs.setdefault(k, [])
                all_outputs[k].append(v)

        return {"hidden_states": hidden_states, "all_outputs": all_outputs}
