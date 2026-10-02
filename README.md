# RSNA Knee Abnormality Detection

A PyTorch research project for the RSNA Knee Abnormality Detection Kaggle competition. The pipeline predicts 12 study-level knee MRI abnormalities from sagittal, coronal, and axial views, using metadata-aware series selection, DICOM preprocessing, slice aggregation, and multi-view fusion.

## What This Project Does

For each MRI study, the pipeline:

1. Finds fluid-sensitive sagittal, coronal, and axial series in the metadata.
2. Selects the strongest available series using sequence description and slice count.
3. Loads, normalizes, resizes, and samples a fixed number of DICOM slices.
4. Encodes slices with a shared image backbone.
5. Aggregates slice features and fuses the three anatomical views.
6. Predicts independent logits for 12 abnormalities:

`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, and `Fracture`.

The modular components are covered by unit tests, while the versioned experiment scripts preserve the evolution of the competition training pipeline.

## Quick Start

Create a virtual environment from the repository root, then install PyTorch for the target hardware and the remaining dependencies:

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

`requirements.txt` is intentionally empty in this checkout, so the explicit install command above is the reliable setup path.

## Data Layout

The training code expects study-level labels, series metadata, and DICOM files in the competition layout. The checked-in CSV files are metadata samples; the DICOM image directories are not included in this repository.

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

Required metadata columns:

- `train.csv`: `StudyInstanceUID` plus all 12 target columns.
- `train_series.csv` and `test_series.csv`: `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, and `Anatomical_Plane`.

The modular dataset drops training rows with missing target labels. Missing anatomical views are represented by zero tensors, allowing studies with incomplete view coverage to be loaded.

## Validate the Installation

From the repository root:

```bash
python -m pytest -q
```

The model tests use small fake backbones where possible, so the test suite does not require the full MRI dataset.

## Run Training on Kaggle

The versioned scripts in `experiments/` are designed for Kaggle's competition directory layout and use CUDA automatically when it is available. The latest workflow is `experiments/train_rsna_knee_v6.py`.

Before a scored offline run:

1. Attach the RSNA competition data to the notebook or execution environment.
2. Attach a dataset containing the matching torchvision pretrained checkpoint.
3. Add the weight dataset path to `CFG.WEIGHT_DIRS` if it is not already searched.
4. Run the selected script from `experiments/` in the Kaggle notebook.

The script intentionally stops when pretrained weights cannot be found. This prevents an offline Kaggle run from silently using a randomly initialized backbone. It writes checkpoints to the configured output directory and produces `submission.csv`.

Example commands from the experiment directory:

```bash
python train_rsna_knee_v6.py
python train_rsna_knee_v6.py --backbone b3 --runs 2 --epochs 7
```

Useful options include `--debug-max-studies`, `--budget-hours`, `--device`, `--labels-csv`, `--image-size`, and `--batch-size`. The script supports offline EfficientNet-B0 and EfficientNet-B3 checkpoints; the checkpoint filename must match the torchvision filename expected by the selected backbone. Use `--no-pretrained` only for debugging, not for a scored submission.

The script can also fall back to the local metadata under `data/` for development, but the corresponding DICOM directories are still required for image loading.

## Experiment Results

The initial Kaggle submission from `arjun.ipynb` achieved an approximate score of **0.487**, below the random-prediction reference point of about 0.50. The main suspected cause was a missing ImageNet checkpoint in the offline Kaggle environment, followed by training with a randomly initialized backbone.

The v2 training script addressed this by requiring pretrained weights to be staged locally before model creation. The 0.487 result should not be treated as a benchmark for the corrected pipeline.

### Reported Kaggle Scores

The following scores were reported from successive experiments. They are leaderboard results, not local validation AUC values.

| Experiment stage | Reported score | Notes |
| --- | ---: | --- |
| Initial `arjun.ipynb` submission | 0.487 | Random or unavailable pretrained backbone suspected |
| Early v4 weak-supervision pipeline | 0.644 | First weak-label and DICOM-cache version |
| v4 with expert-label training | 0.742 | Gold studies added to training |
| v4 gold-label weighting and ensemble update | 0.762 | Improved expert-label supervision |
| v4 validated three-run probability ensemble | **0.763** | Best reported score |
| Five-run logit ensemble experiment | 0.759 | Regression; reverted |
| Repeated three-run experiment | 0.755 | Run-to-run leaderboard variance observed |

Version 5 contains the latest label, validation, caching, and training-stability improvements recorded in the original experiment history. Version 6 extends the offline workflow with improved weak-label handling, denser slice sampling, optional report-label CSVs, and additional training controls; it has not yet received a separate Kaggle leaderboard score.

## Practical Notes

- The competition metric is macro ROC-AUC over the 12 study-level findings.
- `scripts/check_folds.py` reports fold and label distributions from `data/train.csv`.
- Keep DICOM datasets, pretrained weights, checkpoints, caches, and generated predictions outside version control.
- Compare the versioned experiment scripts before selecting a training workflow; they are retained to make changes reproducible.
