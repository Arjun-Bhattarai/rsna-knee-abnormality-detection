import torch
import torch.nn as nn

from .backbone import VisionBackbone
from .aggregation import AttentionAggregator
from .fusion import ViewFusion


class PredictionHead(nn.Module):
    """
    Predict the 12 knee abnormality targets.
    """

    def __init__(
        self,
        feature_dim: int,
        num_targets: int = 12,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, num_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"Expected (B, F), got {x.shape}"
            )

        return self.head(x)


class KneeModel(nn.Module):
    """
    Three-view MRI model.

    Each view is processed using the same vision backbone,
    followed by slice attention, three-view fusion,
    and 12-target prediction.

    Input for each view:
        (B, S, C, H, W)

    Output:
        (B, 12)
    """

    def __init__(
        self,
        backbone: VisionBackbone,
        feature_dim: int,
        num_targets: int = 12,
    ):
        super().__init__()

        self.backbone = backbone
        self.slice_aggregator = AttentionAggregator(feature_dim)
        self.view_fusion = ViewFusion(feature_dim)
        self.prediction_head = PredictionHead(
            feature_dim,
            num_targets,
        )

    def encode_view(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode one MRI view.

        Input:
            (B, S, C, H, W)

        Output:
            (B, F)
        """

        if x.ndim != 5:
            raise ValueError(
                f"Expected (B, S, C, H, W), got {x.shape}"
            )

        batch_size, num_slices, channels, height, width = x.shape

        # Process all slices through the shared backbone.
        x = x.reshape(
            batch_size * num_slices,
            channels,
            height,
            width,
        )

        features = self.backbone(x)

        # Some vision backbones return token sequences.
        if features.ndim == 3:
            features = features.mean(dim=1)

        features = features.reshape(
            batch_size,
            num_slices,
            -1,
        )

        return self.slice_aggregator(features)

    def forward(
        self,
        sagittal: torch.Tensor,
        coronal: torch.Tensor,
        axial: torch.Tensor,
    ) -> torch.Tensor:

        sagittal_features = self.encode_view(sagittal)
        coronal_features = self.encode_view(coronal)
        axial_features = self.encode_view(axial)

        study_features = self.view_fusion(
            sagittal_features,
            coronal_features,
            axial_features,
        )

        return self.prediction_head(study_features)