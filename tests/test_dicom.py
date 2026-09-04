import numpy as np

from src.data.dicom import get_dicom_metadata


def test_get_dicom_metadata():
    class FakeDataset:
        StudyInstanceUID = "study123"
        SeriesInstanceUID = "series123"
        InstanceNumber = 10
        Rows = 640
        Columns = 640

    ds = FakeDataset()

    metadata = get_dicom_metadata(ds)

    assert metadata["study_uid"] == "study123"
    assert metadata["series_uid"] == "series123"
    assert metadata["instance_number"] == 10
    assert metadata["rows"] == 640
    assert metadata["columns"] == 640