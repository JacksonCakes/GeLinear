"""Gemma model config."""

import dataclasses
import torch
from typing import Optional, Sequence


# Keep a mapping from dtype strings to the supported torch dtypes.
_STR_DTYPE_TO_TORCH_DTYPE = dict(
    {
        "float16": torch.float16,
        "float": torch.float32,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
)


@dataclasses.dataclass
class GemmaConfig:
    # The number of tokens in the vocabulary.
    vocab_size: int = 256000
    padding_idx: int = 0
    # The maximum sequence length that this model might ever be used with.
    max_position_embeddings: int = 8192
    # The number of blocks in the model.
    num_hidden_layers: int = 28
    # The number of attention heads used in the attention layers of the model.
    num_attention_heads: int = 16
    # The number of key-value heads for implementing attention.
    num_key_value_heads: int = 16
    # The hidden size of the model.
    hidden_size: int = 3072
    # The dimension of the MLP representations.
    intermediate_size: int = 24576
    # The number of head dimensions.
    head_dim: int = 256
    # The epsilon used by the rms normalization layers.
    rms_norm_eps: float = 1e-6
    # The dtype of the weights.
    dtype: str = "bfloat16"
    # Whether a quantized version of the model is used.
    quant: bool = False
    rope_theta: int = 10000
    # The types of attention used in the layers of the model.
    attention_type: Optional[Sequence[str]] = None
    # The size of the sliding window used for local attention.
    sliding_window_size: Optional[int] = None
    # If provided, the final logits are softcapped to this value.
    final_logit_softcapping: Optional[float] = None
    # If provided, the attention logits are softcapped to this value.
    attn_logit_softcapping: Optional[float] = None

    def get_dtype(self) -> Optional[torch.dtype]:
        """Gets the torch dtype from the config dtype string."""
        return _STR_DTYPE_TO_TORCH_DTYPE.get(self.dtype, None)


def get_model_config() -> GemmaConfig:
    return GemmaConfig(
        num_hidden_layers=42,
        num_attention_heads=16,
        num_key_value_heads=8,
        hidden_size=3584,
        intermediate_size=14336,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        head_dim=256,
        attention_type=["local", "global"] * 21,
        sliding_window_size=4096,
    )
