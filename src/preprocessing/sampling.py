import torch


def uniform_sample_slices(
    volume: torch.Tensor,
    num_slices: int = 32,
) -> torch.Tensor:
    """
    Uniformly sample slices from an MRI volume.

    Input:
        volume: (D, H, W)

    Output:
        sampled volume: (num_slices, H, W)
    """
    if volume.ndim != 3:
        raise ValueError(
            f"Expected volume shape (D, H, W), got {volume.shape}"
        )

    depth = volume.shape[0]

    if depth == 0:
        raise ValueError("Volume contains no slices.")

    indices = torch.linspace(
        0,
        depth - 1,
        steps=num_slices,
    ).round().long()

    return volume[indices]