import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class DINOv2Backbone(nn.Module):
    """
    DINOv2 vision backbone for MRI slices.

    Supports loading from a local Kaggle model path,
    so no internet access is required.
    """

    def __init__(
        self,
        model_name: str = "/kaggle/input/models/metaresearch/dinov2/pytorch/base/1",
        freeze: bool = True,
    ):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            local_files_only=True,
        )

        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=True,
        )

        self.feature_dim = self.model.config.hidden_size

        self.register_buffer(
            "mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1),
        )

        if freeze:
            self.freeze()

    def freeze(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def unfreeze(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: (B, 1, H, W) or (B, 3, H, W)

        Output:
            CLS features: (B, feature_dim)
        """

        if x.ndim != 4:
            raise ValueError(
                f"Expected (B, C, H, W), got {x.shape}"
            )

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        elif x.shape[1] != 3:
            raise ValueError(
                f"Expected 1 or 3 channels, got {x.shape[1]}"
            )

        x = (x - self.mean) / self.std

        outputs = self.model(pixel_values=x)

        return outputs.last_hidden_state[:, 0]