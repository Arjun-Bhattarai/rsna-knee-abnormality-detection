import torch
import torch.nn as nn


class ViewFusion(nn.Module):
    """
    Fuse Sagittal, Coronal, and Axial sequence features.

    Input:
        Each view: (B, F)

    Output:
        Fused study representation: (B, F)
    """

    def __init__(self, feature_dim: int):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(
        self,
        sagittal: torch.Tensor,
        coronal: torch.Tensor,
        axial: torch.Tensor,
    ) -> torch.Tensor:

        if not (
            sagittal.ndim == 2
            and coronal.ndim == 2
            and axial.ndim == 2
        ):
            raise ValueError(
                "All views must have shape (B, F)"
            )

        x = torch.cat(
            [sagittal, coronal, axial],
            dim=1,
        )

        return self.fusion(x)