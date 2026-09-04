import pandas as pd

from src.data.dataset import KneeDataset, TARGETS


def test_dataset_filters_unlabeled_rows(tmp_path):
    rows = [
        {
            "StudyInstanceUID": "study_1",
            **{target: 1 for target in TARGETS},
        },
        {
            "StudyInstanceUID": "study_2",
            **{target: None for target in TARGETS},
        },
    ]

    csv_path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    series_csv = tmp_path / "train_series.csv"
    pd.DataFrame(
        columns=[
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "Fluid_Sensitive",
            "Fat_Suppression",
            "Anatomical_Plane",
        ]
    ).to_csv(series_csv, index=False)

    dataset = KneeDataset(
        csv_path=csv_path,
        series_csv_path=series_csv,
        data_dir=tmp_path,
    )

    assert len(dataset) == 1
    assert len(dataset.labels[0]) == 12