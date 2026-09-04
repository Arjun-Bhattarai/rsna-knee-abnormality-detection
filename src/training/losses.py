import torch
import torch.nn as nn


class WeightedBCELoss(nn.Module):
    """
    Weighted binary cross-entropy loss for the 12 targets.
    """

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()

        self.register_buffer(
            "pos_weight",
            pos_weight.float(),
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        if logits.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: logits={logits.shape}, "
                f"targets={targets.shape}"
            )

        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=self.pos_weight,
        )