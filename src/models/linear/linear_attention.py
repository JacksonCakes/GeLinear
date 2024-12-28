import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.rope import apply_rotary_emb
from src.models.linear.feature_map import HedgehogFeatureMap
from einops import rearrange


def pad(x, chunk_size=64):
    T = x.shape[-2]
    padded_seq_len = ceildiv(T, chunk_size) * chunk_size
    if x.shape[-2] % chunk_size != 0:
        x = F.pad(x, (0, 0, 0, padded_seq_len - T))

    return x


def ceildiv(a, b):
    return -(a // -b)


def chunk_linear_attn(q, k, v, chunk_size=64):
    q, k, v = map(lambda x: pad(x), [q, k, v])
    q = rearrange(q, "b h (n c) d -> b h n c d", c=chunk_size) * (q.shape[-1] ** -0.5)
    k = rearrange(k, "b h (n c) d -> b h n c d", c=chunk_size)
    v = rearrange(v, "b h (n c) d -> b h n c d", c=chunk_size)
    kv = k.transpose(-1, -2) @ v
    kv = kv.cumsum(2)
    kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2)
    inter = q @ kv  # (b, h, n, c, d) @ (b, h, n, d, d) -> (b, h, n, c, d)
    intra = (
        (q @ k.transpose(-1, -2)).masked_fill_(
            torch.triu(
                torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device),
                diagonal=1,
            ),
            0,
        )
    ) @ v
    o = inter + intra
    return rearrange(o, "b h n c d -> b h (n c) d")


class GemmaLinearAttention(nn.Module):
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
        self.feature_map_q = HedgehogFeatureMap(
            num_heads=num_heads,
            head_dim=head_dim,
            feature_dim=64,
        )
        self.feature_map_k = HedgehogFeatureMap(
            num_heads=num_heads,
            head_dim=head_dim,
            feature_dim=64,
        )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling_factor = head_dim**-0.5
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_group = self.num_heads // self.num_kv_heads
        self.attn_type = attn_type
        self.sliding_window_size = sliding_window_size
        self.attn_logit_softcapping = attn_logit_softcapping

    def forward(self, hidden_states, freqs_cis, mask, output_attn):
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
        output = chunk_linear_attn(q, k, v)
        output = (
            output.transpose(1, 2)
            .contiguous()[:, :seq_len, ...]
            .view(batch_size, seq_len, -1)
        )
        output = self.o_proj(output)
        return {"hidden_states": output}
