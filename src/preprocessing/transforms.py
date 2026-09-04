import torch
import torch.nn.functional as F


def resize_volume(
    volume: torch.Tensor,
    size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """
    Resize every slice of an MRI volume.

    Input:
        volume: (D, H, W)

    Output:
        resized volume: (D, 224, 224)
    """
    if volume.ndim != 3:
        raise ValueError(
            f"Expected volume shape (D, H, W), got {volume.shape}"
        )

    volume = volume.float()

    # Add channel dimension: (D, 1, H, W)
    volume = volume.unsqueeze(1)

    volume = F.interpolate(
        volume,
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    # Remove channel dimension
    return volume.squeeze(1)