import numpy as np
from sklearn.metrics import roc_auc_score


def macro_roc_auc(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """
    Calculate macro-averaged ROC-AUC across all targets.

    Args:
        targets: Ground-truth labels, shape (N, 12)
        predictions: Prediction scores, shape (N, 12)

    Returns:
        Macro ROC-AUC.
    """

    if targets.shape != predictions.shape:
        raise ValueError(
            f"Shape mismatch: targets={targets.shape}, "
            f"predictions={predictions.shape}"
        )

    scores = []

    for i in range(targets.shape[1]):
        # AUC is undefined when a validation fold
        # contains only one class.
        if len(np.unique(targets[:, i])) < 2:
            continue

        scores.append(
            roc_auc_score(
                targets[:, i],
                predictions[:, i],
            )
        )

    if not scores:
        raise ValueError(
            "ROC-AUC cannot be calculated: "
            "no target contains both classes."
        )

    return float(np.mean(scores))