# RSNA Knee Abnormality Detection

A PyTorch research codebase for multi-label abnormality detection from knee MRI studies. The model uses three anatomical views (sagittal, coronal, and axial), selects fluid-sensitive series from DICOM metadata, aggregates slices with attention, and predicts 12 abnormalities for each study.

## Project Status

The reusable dataset, preprocessing, model, loss, metric, and trainer components are implemented under `src/` and covered by tests under `tests/`. The current end-to-end training entry point is `train_rsna_knee_v2.py`, which is designed for an offline Kaggle submission environment.

The files in `scripts/` and `configs/` are scaffolding for a future modular CLI; they are currently empty and should not be treated as runnable commands yet.

## Model Overview

For each study, the pipeline:

1. Filters series metadata to fluid-sensitive sagittal, coronal, and axial acquisitions.
2. Selects the best available series using sequence description and slice count.
3. Loads and normalizes DICOM slices, then samples a fixed number of slices per view.
4. Encodes slices with a shared vision backbone.
5. Aggregates slices with attention and fuses the three views.
6. Produces independent logits for the following targets:

`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, and `Fracture`.

## Repository Layout

```text
src/
	data/            DICOM loading, study dataset, and fold splitting
	preprocessing/   normalization, sampling, and transforms
	models/          backbone, slice aggregation, view fusion, and model
	training/        losses and training loop
	inference/       prediction utilities
	utils/           metrics, logging, seeding, and configuration helpers
tests/             unit tests for the implemented modules
train_rsna_knee_v2.py
									 standalone offline/Kaggle training and inference script
data/              local CSV metadata included with this checkout
weights/           location for local model weights
outputs/           checkpoints and generated artifacts
```

## Requirements

Python 3.10 or newer and a CPU or CUDA-enabled PyTorch installation are recommended. `requirements.txt` is currently empty, so install the runtime and test dependencies explicitly in a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch torchvision pandas numpy scikit-learn pydicom pytest pyyaml tqdm
```

For CUDA, install the PyTorch build that matches the target machine from the [official PyTorch selector](https://pytorch.org/get-started/locally/) before installing the remaining packages.

## Data Layout

The dataset class expects study-level labels, series metadata, and DICOM files arranged like this:

```text
dataset/
	train.csv
	test.csv
	train_series.csv
	test_series.csv
	train_series/
		<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
	test_series/
		<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
```

`train.csv` must contain `StudyInstanceUID` plus all 12 target columns. The series CSV files must include `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, and `Anatomical_Plane`. The checked-in CSV files under `data/` are metadata examples; the DICOM image directories are not included in this repository.

## Run the Tests

From the repository root:

```bash
python -m pytest -q
```

The model tests use small fake backbones where possible, so the test suite does not require the full MRI dataset.

## Kaggle Training

`train_rsna_knee_v2.py` is configured for the RSNA competition directory layout and automatically uses CUDA when available. Before a scored offline run:

1. Attach the RSNA competition data to the notebook or execution environment.
2. Attach a dataset containing the matching torchvision pretrained checkpoint.
3. Add that dataset path to `Config.PRETRAINED_WEIGHTS_DIRS` if it is not one of the default search paths.
4. Run the script in the Kaggle notebook or copy it into the submission notebook.

The script intentionally stops when pretrained weights cannot be found. This avoids silently training a randomly initialized backbone when internet access is disabled. It writes checkpoints and the final submission according to `Config.OUTPUT_DIR` and `Config.SUBMISSION_PATH`.

The standalone script supports offline EfficientNet-B0 and ResNet-34 checkpoints. The checkpoint filename must match the torchvision filename expected by the selected backbone.

## Development Notes

- Missing or unavailable views are represented by zero tensors in `KneeDataset`.
- The modular dataset currently drops rows with missing target labels.
- The training criterion and metric operate on the 12-target multi-label output.
- Keep large DICOM datasets, checkpoints, and generated predictions outside version control.
- The modular training and inference command wiring still needs to be completed before `scripts/train.py` and `scripts/inference.py` can be used as CLI entry points.
