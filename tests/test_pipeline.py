import torch

from src.preprocessing.pipeline import preprocess_volume


def test_preprocess_volume():
    volume = torch.randn(40, 640, 640)

    processed = preprocess_volume(
        volume,
        num_slices=32,
        image_size=(224, 224),
    )

    assert processed.shape == (32, 224, 224)
    assert processed.dtype == torch.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0