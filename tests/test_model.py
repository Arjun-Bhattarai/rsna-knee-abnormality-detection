import torch
import torch.nn as nn

from src.models.backbone import VisionBackbone
from src.models.model import KneeModel


class FakeBackbone(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.projection = nn.Linear(224 * 224, feature_dim)

    def forward(self, x):
        x = x.flatten(1)
        return self.projection(x)


def test_knee_model():
    feature_dim = 128

    backbone = VisionBackbone(
        backbone=FakeBackbone(feature_dim),
        feature_dim=feature_dim,
        freeze=False,
    )

    model = KneeModel(
        backbone=backbone,
        feature_dim=feature_dim,
        num_targets=12,
    )

    sagittal = torch.randn(2, 8, 1, 224, 224)
    coronal = torch.randn(2, 8, 1, 224, 224)
    axial = torch.randn(2, 8, 1, 224, 224)

    output = model(
        sagittal,
        coronal,
        axial,
    )

    assert output.shape == (2, 12)