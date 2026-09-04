import numpy as np
import torch

from .normalization import percentile_normalize
from .transforms import resize_volume
from .sampling import uniform_sample_slices


def preprocess_volume(
    volume: np.ndarray | torch.Tensor,
    num_slices: int = 32,
    image_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """
    Complete MRI preprocessing pipeline.

    Input:
        volume: (D, H, W), NumPy array or torch.Tensor

    Output:
        processed volume: (num_slices, H, W)
    """

    # Convert to NumPy for intensity normalization
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()

    if not isinstance(volume, np.ndarray):
        raise TypeError(
            f"Expected NumPy array or torch.Tensor, got {type(volume)}"
        )

    # Normalize intensity
    volume = percentile_normalize(volume)

    # Convert to tensor
    volume = torch.from_numpy(volume)

    # Resize each slice
    volume = resize_volume(
        volume,
        size=image_size,
    )

    # Sample fixed number of slices
    volume = uniform_sample_slices(
        volume,
        num_slices=num_slices,
    )

    return volume