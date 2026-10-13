# RSNA Knee Abnormality Detection

This repository contains a PyTorch research pipeline for the RSNA Knee Abnormality Detection competition. It predicts twelve study-level findings from sagittal, coronal, and axial knee MRI series: ACL, MCL, medial meniscus, lateral meniscus, medial OA, lateral OA, PF OA, effusion, synovitis, Baker's cyst, contusion, and fracture.

The project combines metadata-aware series selection, DICOM loading, intensity normalization, fixed-depth slice sampling, shared image encoding, attention-based slice aggregation, three-view fusion, and multi-label prediction. The modular implementation is kept under `src/`, while the versioned competition workflows are kept under `experiments/`.

## Current workflow

The recommended competition workflow is version 6 in `experiments/train_rsna_knee_v6.py`. It is designed for an offline Kaggle GPU session and uses locally attached pretrained EfficientNet weights rather than attempting to download weights during execution. It also supports report-derived labels, expert-label overrides, cached DICOM preprocessing, repeated training runs, test-time augmentation, and optional rank blending with an independent reader prediction file.

The pipeline expects the competition metadata and DICOM directories to be available in the standard RSNA layout. The repository includes small metadata samples for development, but the full image data and pretrained weights are intentionally kept outside version control. Missing anatomical views are handled as empty views, and malformed or unreadable series are skipped safely so that one damaged series does not terminate a complete run.

The competition metric is macro ROC-AUC across all twelve findings. Because the expert-labelled subset is small, local validation scores should be interpreted cautiously and should not be treated as a direct replacement for leaderboard results.

## Reported scores

The scores below are selected Kaggle public leaderboard milestones. They show the progression of the project without listing every failed run or small run-to-run fluctuation.

| Milestone | Public score | What changed |
| --- | ---: | --- |
| Initial submission | 0.487 | Baseline with an unavailable or randomly initialized backbone |
| First weak-supervision pipeline | 0.644 | Report-derived labels and reusable DICOM caching |
| Expert-label and ensemble improvements | 0.763 | Stronger expert supervision and validated probability ensembling |
| Best early notebook result | 0.879 | Later notebook iteration with improved training and inference |
| **Best recorded result** | **0.882** | Kaggle notebook Version 49 |

The best recorded public score is **0.882**, a substantial improvement over the initial 0.487 submission and the earlier 0.763 milestone. The latest checked-in workflow is version 6 in `experiments/train_rsna_knee_v6.py`; Kaggle notebook revision numbers do not map one-to-one to repository script versions.

## Version history and score progression

Version 1 was the original competition attempt. It relied on a backbone that could fall back to random initialization when pretrained weights were unavailable in Kaggle's offline environment. That submission reached approximately 0.487, below the random-prediction reference point of about 0.50. The main lesson was that reliable offline weight loading was more important than further hyperparameter tuning.

Version 2 made offline pretrained weights mandatory and stopped instead of silently training a random backbone. It also introduced real augmentation, explicit missing-label handling, per-target class weighting, differential learning rates, warmup and cosine scheduling, more reliable series ordering, additional slices, gradient clipping, early stopping, test-time augmentation, and weighted fold ensembling. These changes established the foundation for meaningful training.

Version 3 refined the pretrained multi-view training workflow and improved the handling of the competition's incomplete labels. Its main limitation was that the majority of studies still contributed little or no useful training signal because only 58 of the 4,407 training studies had expert labels. The resulting validation and submission behaviour showed that improving label coverage was the next major opportunity.

Version 4 introduced weak supervision for the previously unlabeled studies. It generated soft labels from multilingual radiology reports using keyword matching, clause-level negation, and side attribution. It also decoded each DICOM series once into a reusable uint8 cache, normalized laterality so medial and lateral findings remained spatially consistent, removed unsafe horizontal flips, and used the expert studies for checkpoint selection. This was the first major score improvement, with reported leaderboard results reaching approximately 0.644 and later 0.742 and 0.762 as expert weighting and ensembling were improved.

Version 5 consolidated the strongest version 4 recipe and corrected several operational issues. It retained the validated weak-label parser, expert-label weighting, probability averaging, and repeated-run ensemble. It separated checkpoint selection from training labels to reduce validation leakage, fixed a study-ID series-plan failure, reduced unnecessary cache flushing, and made larger hidden-test cache creation safer. The best reported version 5 result was approximately 0.763 from a validated three-run probability ensemble. Additional five-run and repeated-run experiments scored approximately 0.759 and 0.755, demonstrating that more runs do not automatically improve macro ROC-AUC.

Version 6 focuses on improving the quality and reliability of weak supervision rather than simply increasing model size. It prefers an attached hybrid or consensus report-label CSV when available, keeps expert labels as hard overrides, assigns a low-confidence prior to unmentioned findings instead of treating silence as a confirmed negative, uses denser 2.5D slice windows, and adds masked multi-instance aggregation for focal abnormalities. It also adds more controlled augmentation, a global fusion head, optional EfficientNet-B3 support, configurable slice sampling, configurable MIL blending, stronger runtime safeguards, and optional reader rank blending. Version 6 has not yet received a separate leaderboard score, so these changes should be evaluated through reproducible submissions rather than assumed to improve the score.

## Changes most likely to improve the score

The highest-impact improvement is ensuring that the backbone is initialized from the correct offline ImageNet checkpoint. A missing checkpoint can reduce the entire submission to near-random performance, regardless of the training schedule or model head.

The next major improvement is increasing the quality and coverage of training labels. The expert set contains only 58 studies, so a carefully validated hybrid or consensus report-label dataset is more valuable than treating the remaining reports as unlabeled. External labels should be checked for study-ID coverage, target-column completeness, finite values, and compatibility with the competition test distribution. Expert labels should always override weak or external labels.

Weak labels should express uncertainty. Explicit positive and negative report evidence can receive stronger weights, ambiguous language should receive a much smaller weight, and unmentioned findings should not automatically become hard negatives. This reduces label noise and is especially important for rare findings such as ACL injury, fracture, synovitis, and contusion.

Laterality and anatomical view consistency are also critical. Horizontal flips can exchange medial and lateral abnormalities, which directly harms four target columns. Selecting complementary fluid-sensitive series, ordering slices consistently, and normalizing right-knee orientation are safer improvements than generic image augmentation.

Caching the decoded DICOM data improves both speed and experiment quality. Faster epochs allow more controlled runs, expert-aware checkpoint selection, and probability ensembling within the GPU time limit. The cache is an efficiency improvement, but it becomes a score improvement only when the recovered time is used for better validation, stronger labels, or genuinely complementary models.

Repeated probability ensembles are more reliable than blindly increasing the number of runs. The reported results show that a validated three-run ensemble outperformed the tested five-run and repeated variants. Run diversity, independent validation, and stable checkpoint selection matter more than ensemble size alone.

An independent reader model can provide complementary signal when its predictions are complete and validated against every test study. Version 6 can blend reader predictions by per-target rank, which is appropriate for an AUC-based competition metric. The blend weight should be treated as an experiment parameter and checked against out-of-fold evidence; a reader model that is highly correlated with the image model may add little.

## Repository guidance

The `src/` package contains reusable preprocessing, data, model, loss, metric, and training components. The `experiments/` directory preserves the chronological competition workflows so that earlier assumptions and reported results remain reproducible. The `tests/` directory covers the modular components and important edge cases, including malformed inputs, missing series, slice sampling, model shapes, and metric handling.

Full DICOM data, pretrained weights, generated caches, checkpoints, and submissions should remain outside version control. The current modular test suite is designed to run without the full competition dataset or a GPU.

Reported leaderboard values in this document are historical observations from the project and are not guarantees of future performance. Any new score should be recorded together with the exact version, label source, backbone, image settings, number of runs, ensemble method, and validation procedure.
