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


def test_dataset_handles_missing_series_gracefully(tmp_path):
    rows = [{
        "StudyInstanceUID": "study_1",
        **{target: 1 for target in TARGETS},
    }]

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

    item = dataset[0]
    assert len(item) == 4
    assert item[0].shape == (32, 1, 224, 224)
    assert item[1].shape == (32, 1, 224, 224)
    assert item[2].shape == (32, 1, 224, 224)
    assert item[3].shape == (12,)