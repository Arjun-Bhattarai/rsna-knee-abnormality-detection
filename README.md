# RSNA Knee Abnormality Detection

A PyTorch research codebase for the RSNA Knee Abnormality Detection Kaggle competition. It performs multi-label abnormality detection from knee MRI studies using three anatomical views (sagittal, coronal, and axial), fluid-sensitive series selection, attention-based slice aggregation, and 12 abnormality predictions for each study.

## Project Status

This repository contains the training and experiment code for a Kaggle competition submission. The reusable dataset, preprocessing, model, loss, metric, and trainer components are implemented under `src/` and covered by tests under `tests/`. The current end-to-end training entry point is `train_rsna_knee_v2.py`, which is designed for an offline Kaggle submission environment.

The only auxiliary script currently retained is `scripts/check_folds.py`, which reports fold and label distributions from `data/train.csv`. The end-to-end training entry point remains `train_rsna_knee_v2.py`.

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
	utils/           metrics and related helpers
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

## Experiment Results

The initial Kaggle submission from `arjun.ipynb` achieved an approximate score of **0.487**. This was below the random-prediction reference point of about 0.50 and indicated that the model was not learning useful signal. The main suspected cause was the scored notebook failing to download ImageNet pretrained weights because Kaggle execution has no internet access, then continuing with a randomly initialized backbone.

The v2 training script addresses this by requiring pretrained weights to be staged locally before model creation. Do not treat the 0.487 result as a final benchmark for the corrected pipeline; record a new Kaggle score after running the offline-weight workflow.

## Development Notes

- Missing or unavailable views are represented by zero tensors in `KneeDataset`.
- The modular dataset currently drops rows with missing target labels.
- The training criterion and metric operate on the 12-target multi-label output.
- Keep large DICOM datasets, checkpoints, and generated predictions outside version control.
- `scripts/check_folds.py` is a diagnostic utility; training and inference use `train_rsna_knee_v2.py`.
