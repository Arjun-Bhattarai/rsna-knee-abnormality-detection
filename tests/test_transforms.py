import torch

from src.preprocessing.transforms import resize_volume


def test_resize_volume():
    volume = torch.randn(20, 640, 640)

    resized = resize_volume(volume, (224, 224))

    assert resized.shape == (20, 224, 224)
    assert resized.dtype == torch.float32