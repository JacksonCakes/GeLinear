import torch
import torch.nn.functional as F
from einops import rearrange


def quadratic_attention(q, k, v, eps=1e-12):
    a = torch.einsum("bhmd,bhnd->bhmn", q, k)
    m, n = a.shape[-2:]
    causal_mask = torch.ones((m, n), device=a.device, dtype=torch.bool).triu(n - m + 1)
    a_pred = a.masked_fill(causal_mask, 0)
    # Normalize attention scores
    scores = a_pred / (a_pred.sum(dim=-1, keepdim=True) + eps)
    attn_output = torch.einsum("bhmn,bhnd->bhmd", scores, v)
    return attn_output, scores


def softmax_attention(q, k, v, mask, attn_logit_softcapping, scaling_factor):
    qk = torch.einsum("bhmd,bhnd->bhmn", q, k)
    qk.mul_(scaling_factor)
    # logits ← soft_cap ∗ tanh(logits/soft_cap)
    if attn_logit_softcapping is not None:
        qk = qk / attn_logit_softcapping
        qk = torch.tanh(qk)
        qk = qk * attn_logit_softcapping

    scores = qk + mask
    scores = F.softmax(scores.float(), dim=-1).type_as(q)
    # [batch_size, num_heads, seq_len, head_dim]
    attn_output = torch.einsum("bhmn,bhnd->bhmd", scores, v)
    return attn_output, scores


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
