import numpy as np


def percentile_normalize(
    volume: np.ndarray,
    lower: float = 1.0,
    upper: float = 99.0,
) -> np.ndarray:
    """
    Normalize an MRI volume using percentile clipping.

    Args:
        volume: MRI volume, shape (D, H, W)
        lower: Lower intensity percentile.
        upper: Upper intensity percentile.

    Returns:
        Normalized volume in [0, 1], float32.
    """
    volume = volume.astype(np.float32)

    low = np.percentile(volume, lower)
    high = np.percentile(volume, upper)

    if high <= low:
        return np.zeros_like(volume, dtype=np.float32)

    volume = np.clip(volume, low, high)
    volume = (volume - low) / (high - low)

    return volume.astype(np.float32)