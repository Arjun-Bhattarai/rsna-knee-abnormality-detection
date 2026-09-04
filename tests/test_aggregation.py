import torch

from src.models.aggregation import AttentionAggregator


def test_attention_aggregation():
    model = AttentionAggregator(feature_dim=128)

    x = torch.randn(4, 32, 128)

    output = model(x)

    assert output.shape == (4, 128)