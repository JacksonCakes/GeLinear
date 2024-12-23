import torch
import torch.nn as nn


class HedgehogFeatureMap(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,  # input dim
        feature_dim: int,  # output dim
        # dtype: torch.dtype,
        # device: torch.device,
        bias: bool = False,
        eps: float = 1e-12,
    ):
        super().__init__()

        self.layer = nn.Parameter(
            torch.zeros(
                (num_heads, head_dim, feature_dim),
                # dtype=dtype,
                # device=device,
            )
        )
        nn.init.kaiming_uniform_(self.layer)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    (1, num_heads, 1, 1),
                    # dtype=dtype,
                    # device=device,
                )
            )
            nn.init.kaiming_uniform_(self.bias)
        else:
            self.bias = 0.0  # hack
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        x = (batch_size, num_heads, seq_len, head_dim)
        """
        output = torch.einsum("hdf,bhld->bhlf", self.layer, x) + self.bias
        output = torch.cat(
            [torch.softmax(output, dim=-1), torch.softmax(-output, dim=-1)], dim=-1
        ).clamp(min=self.eps)
        return output


class TrainableHedgehog(nn.Module):
    def __init__(
        self, num_feature_maps, num_heads, head_dim, feature_dim
    ):  # , dtype, device):
        super(TrainableHedgehog, self).__init__()
        self.scaling_factor = head_dim**-0.5
        self.feature_maps_q = nn.ModuleList(
            [
                HedgehogFeatureMap(
                    num_heads=num_heads,
                    head_dim=head_dim,
                    feature_dim=feature_dim,
                )
                for _ in range(num_feature_maps)
            ]
        )
        self.feature_maps_k = nn.ModuleList(
            [
                HedgehogFeatureMap(
                    num_heads=num_heads,
                    head_dim=head_dim,
                    feature_dim=feature_dim,
                )
                for _ in range(num_feature_maps)
            ]
        )

    def forward(self, q, k, v, layer_idx, eps=1e-12):
        # Forward pass for a specific layer
        phi_q = self.feature_maps_q[layer_idx](q)
        phi_k = self.feature_maps_k[layer_idx](k)
        a = torch.einsum("bhmd,bhnd->bhmn", phi_q, phi_k)
        m, n = a.shape[-2:]
        causal_mask = torch.ones((m, n), device=a.device, dtype=torch.bool).triu(
            n - m + 1
        )
        a_pred = a.masked_fill(causal_mask, 0)
        # Normalize attention scores
        a_pred = a_pred / (a_pred.sum(dim=-1, keepdim=True) + eps)
        out_pred = torch.einsum("bhmn,bhnd->bhmd", a_pred, v)
        return out_pred
