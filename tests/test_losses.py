import torch

from src.training.losses import WeightedBCELoss


def test_weighted_bce_loss():
    pos_weight = torch.ones(12)

    loss_fn = WeightedBCELoss(pos_weight)

    logits = torch.randn(4, 12)
    targets = torch.randint(0, 2, (4, 12)).float()

    loss = loss_fn(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0