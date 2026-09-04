import torch
import torch.nn as nn

from src.models.backbone import VisionBackbone


def test_backbone_freeze_unfreeze():
    backbone = nn.Linear(10, 8)

    model = VisionBackbone(
        backbone=backbone,
        feature_dim=8,
        freeze=True,
    )

    assert all(
        not p.requires_grad
        for p in model.parameters()
    )

    model.unfreeze()

    assert all(
        p.requires_grad
        for p in model.parameters()
    )


def test_backbone_forward():
    backbone = nn.Linear(10, 8)

    model = VisionBackbone(
        backbone=backbone,
        feature_dim=8,
        freeze=True,
    )

    x = torch.randn(4, 10)
    output = model(x)

    assert output.shape == (4, 8)