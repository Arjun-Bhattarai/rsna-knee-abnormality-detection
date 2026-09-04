import torch

from src.preprocessing.sampling import uniform_sample_slices


def test_uniform_sample_slices():
    volume = torch.randn(80, 224, 224)

    sampled = uniform_sample_slices(volume, num_slices=32)

    assert sampled.shape == (32, 224, 224)
    