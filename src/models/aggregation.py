import torch
import torch.nn as nn


class AttentionAggregator(nn.Module):
    """
    Aggregate slice-level features into one sequence-level feature.

    Input:
        (batch, slices, feature_dim)

    Output:
        (batch, feature_dim)
    """

    def __init__(self, feature_dim: int):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected (B, S, F), got {x.shape}"
            )

        scores = self.attention(x)

        weights = torch.softmax(scores, dim=1)

        aggregated = torch.sum(
            weights * x,
            dim=1,
        )

        return aggregated