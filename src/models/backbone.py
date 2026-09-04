import torch
import torch.nn as nn


class VisionBackbone(nn.Module):
    """
    Wrapper around a pretrained vision encoder.

    The actual pretrained model will be injected later,
    keeping the rest of our architecture independent
    from the specific backbone.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        freeze: bool = True,
    ):
        super().__init__()

        self.backbone = backbone
        self.feature_dim = feature_dim

        if freeze:
            self.freeze()

    def freeze(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)