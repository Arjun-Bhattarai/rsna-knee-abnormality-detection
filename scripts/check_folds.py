import pandas as pd

from src.data.dataset import TARGETS
from src.data.splits import create_multilabel_folds


TRAIN_CSV = "data/train.csv"


df = pd.read_csv(TRAIN_CSV)

folds = create_multilabel_folds(df)

print("Labeled studies:", len(folds))
print("\nStudies per fold:")
print(folds["fold"].value_counts().sort_index())

print("\nPositive labels per fold:")
print(
    folds.groupby("fold")[TARGETS]
    .sum()
    .T
)