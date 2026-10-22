# RSNA Knee Abnormality Detection:

A PyTorch pipeline for predicting **12 knee abnormalities** from MRI studies using multi-view imaging, report-derived weak labels, and attention-based feature fusion.

## Results at a Glance

**Best public ROC-AUC: 0.882 — Kaggle Version 49**  
Latest listed successful run: **0.881 — Version 52**

| Milestone | Public ROC-AUC | Change | What the milestone represents |
|---|---:|---:|---|
| Early baseline — V9 | 0.574 | — | Initial low-scoring experiments established a baseline. |
| V14 | 0.644 | **+0.070** vs V9 | First clear improvement; project history associates this phase with report-derived weak labels and DICOM caching. |
| V22 | 0.742 | **+0.098** vs V14 | A major improvement in the next experiment phase; the score log alone does not identify the exact code change. |
| V24 | 0.763 | **+0.021** vs V22 | Stronger result; this phase has been associated with improved expert-label use and probability ensembling. |
| V45 | 0.879 | **+0.116** vs V24 | Largest later jump. The available run log does not prove which individual change caused it. |
| **V49 — Best** | **0.882** | **+0.003** vs V45 | Best recorded public score. |
| V52 — Latest listed | 0.881 | −0.001 vs V49 | Remained close to the best result, but did not surpass it. |

From V9's 0.574 to the best score of 0.882, the absolute gain was **0.308 ROC-AUC points**. From the earliest reported score of 0.487 to 0.882, the gain was **0.395 points**.

## How the Approach Evolved

The experiment history shows several distinct phases. The score log records outcomes, but does not contain a complete change log for every notebook revision, so causal explanations below are only included where the available project notes support them.

1. **Baseline experiments (V7–V13; recorded scores include 0.574 in V9).** Early submissions established a starting point, with scores varying as the pipeline developed.
2. **Weak-supervision improvement (V14: 0.644).** Project notes associate this phase with report-derived labels and reusable DICOM preprocessing/cache work. These can increase the amount of usable training signal and reduce repeated preprocessing, though the score alone cannot isolate their individual impact.
3. **Steady model and training iteration (V22–V35: roughly 0.728–0.763).** The score reached 0.742 at V22 and 0.763 at V24, then fluctuated around the mid-0.7s. Notes associate this phase with better use of expert labels and probability ensembling; exact per-version diffs are not available in the score log.
4. **Major later improvement (V45: 0.879).** This is the largest documented jump from the V24 milestone. The available history does not establish which specific architecture, data, or inference change caused it, so it should be described as a major experimental improvement rather than attributed to an unverified technique.
5. **Fine-tuning near the peak (V46–V52: 0.877–0.882).** Results became comparatively stable near 0.88. V49 reached 0.882; V52 scored 0.881. Small differences may reflect experiment changes or run variation, but the supplied log does not establish the cause.

## Technical Approach

- **Multi-view MRI:** selects sagittal, coronal, and axial series using DICOM metadata.
- **2.5D slice processing:** encodes neighboring MRI slices with a pretrained EfficientNet backbone.
- **Attention-based aggregation:** combines slice features and fuses information across available views.
- **Weak supervision:** derives labels from radiology reports, including handling of negation and laterality; expert labels take priority where available.
- **Training and inference tools:** memory-mapped DICOM caching, confidence-weighted loss, mixed precision, test-time augmentation, and probability ensembling are described in the project notes.

## Pipeline

`DICOM studies → series/view selection → preprocessing and cache → 2.5D slices → image encoder → slice/view fusion → 12 predictions`

## Repository Layout

- `experiments/` — versioned experiment workflows
- `experiments/train_rsna_knee_v6.py` — latest checked-in workflow identified in the project notes
- `src/` — reusable data, preprocessing, model, training, and metrics code
- `tests/` — tests for components and edge cases

**Important:** Kaggle notebook versions (such as V49) are not the same as repository script versions (such as v6). The available material does not prove that the current v6 workflow produced the best V49 score. To document exact per-version changes confidently, compare the saved Kaggle notebooks or source diffs and record the change alongside each score.

## Reproducibility Note

The workflow is intended for a Kaggle GPU environment and depends on the competition DICOM data and the expected pretrained checkpoint and label files. Dependencies and a fully verified end-to-end run are not established by the available notes. This is a research project, not a clinically validated diagnostic system.
