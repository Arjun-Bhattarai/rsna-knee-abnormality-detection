import pandas as pd

from src.data.dataset import TARGETS
from src.data.splits import create_multilabel_folds


def test_multilabel_folds():
    rows = []

    for i in range(20):
        rows.append({
            "StudyInstanceUID": f"study_{i}",
            **{
                target: (i + j) % 2
                for j, target in enumerate(TARGETS)
            },
        })

    df = pd.DataFrame(rows)

    result = create_multilabel_folds(
        df,
        n_splits=5,
    )

    assert "fold" in result.columns
    assert result["fold"].nunique() == 5
    assert len(result) == 20
    assert not (result["fold"] == -1).any()