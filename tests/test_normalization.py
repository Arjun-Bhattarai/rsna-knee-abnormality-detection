import numpy as np

from src.preprocessing.normalization import percentile_normalize


def test_percentile_normalize():
    volume = np.arange(100, dtype=np.float32).reshape(10, 10)

    normalized = percentile_normalize(volume)

    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0