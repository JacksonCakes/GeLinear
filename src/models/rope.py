import torch
import torch.nn as nn
from typing import Optional

torch.manual_seed(42)  # For reproducibility


@torch.compiler.disable()
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the frequency cis."""

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


@torch.compiler.disable()
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the query and key tensors."""
    x_ = torch.view_as_complex(
        torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1)
    )
    x_out = torch.view_as_real(x_ * freqs_cis).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(
        1, 2
    )
    return x_out


class RotaryEmbedding(nn.Module):
    def __init__(self, theta: float, head_dim: int, max_seq_len: int):
        super().__init__()
        self.theta = theta
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim=head_dim, end=max_seq_len, theta=theta),
            persistent=False,
        )

    def forward(
        self,
        seq_len: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
    ):
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seq_len is not None:
            return self.freqs_cis[0:seq_len]
