import torch

from src.models.fusion import ViewFusion


def test_view_fusion():
    model = ViewFusion(feature_dim=128)

    sagittal = torch.randn(4, 128)
    coronal = torch.randn(4, 128)
    axial = torch.randn(4, 128)

    output = model(
        sagittal,
        coronal,
        axial,
    )

    assert output.shape == (4, 128)