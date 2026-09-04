import torch

from .normalization import percentile_normalize
from .transforms import resize_volume
from .sampling import uniform_sample_slices


def preprocess_volume(
    volume: torch.Tensor,
    num_slices: int = 32,
    image_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """
    Complete MRI preprocessing pipeline.

    Input:
        volume: (D, H, W)

    Output:
        processed volume: (num_slices, H, W)
    """

    # Normalize intensity
    volume = torch.from_numpy(
        percentile_normalize(volume.numpy())
    )

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