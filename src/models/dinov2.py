import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class DINOv2Backbone(nn.Module):
    """
    DINOv2 vision backbone.

    Converts single-channel MRI slices to 3-channel images
    and extracts a study-independent feature vector.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        freeze: bool = True,
    ):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(
            model_name
        )

        self.model = AutoModel.from_pretrained(
            model_name
        )

        self.feature_dim = self.model.config.hidden_size

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

        outputs = self.model(pixel_values=x)

        return outputs.last_hidden_state[:, 0]
    