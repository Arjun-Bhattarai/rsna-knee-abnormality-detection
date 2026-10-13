import numpy as np

from src.utils.metrics import macro_roc_auc


def test_macro_roc_auc():
    targets = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 1, 0],
        [1, 1, 1],
    ])

    predictions = np.array([
        [0.1, 0.1, 0.2],
        [0.2, 0.8, 0.7],
        [0.8, 0.7, 0.3],
        [0.9, 0.9, 0.8],
    ])

    score = macro_roc_auc(targets, predictions)

    assert 0.0 <= score <= 1.0


def test_macro_roc_auc_ignores_nan_values():
    targets = np.array([
        [0.0, 1.0],
        [1.0, np.nan],
        [0.0, 0.0],
        [1.0, 1.0],
    ])
    predictions = np.array([
        [0.1, 0.9],
        [0.8, np.nan],
        [0.3, 0.2],
        [0.7, 0.8],
    ])

    score = macro_roc_auc(targets, predictions)

    assert 0.0 <= score <= 1.0