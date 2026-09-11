"""
RSNA Knee Abnormality Detection — v2
=====================================

WHY YOUR SCORE WAS 0.487 (worse than random guessing = 0.5):
--------------------------------------------------------------
Your original code called `models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)`
inside the SAME script that runs during the scored Kaggle submission. Kaggle submission
runs have internet DISABLED. That download silently fails, the except-block falls back to
`weights=None` (i.e. a RANDOMLY INITIALIZED backbone), and your model then trains for only
4 epochs from scratch on medical images with no ImageNet priors. That is functionally
equivalent to random predictions -> AUC ~0.5. Everything else you tune is irrelevant until
this is fixed.

THE FIX (two parts, both implemented below):
1. `stage_offline_pretrained_weights()` finds pretrained weight files in an ATTACHED
    Kaggle Dataset before the model is built, and the model loads them directly without
    any network access.
   -> You must attach a Kaggle Dataset containing the correct .pth file(s). See the
      docstring on that function for exact filenames and how to create the dataset.
2. If no local weights are found, the code stops instead of silently training a random
    backbone and producing a misleading submission.

OTHER FIXES INCLUDED:
- Real train-time augmentation (flip / small rotation / intensity jitter) — previously
  train and val used identical (i.e. zero) augmentation.
- Missing labels are now tracked with a mask and excluded from the loss instead of being
  silently treated as "confirmed negative" via fillna(0.0).
- `pos_weight` computed per-fold and passed to BCEWithLogitsLoss to fight class imbalance
  (rare abnormalities were previously drowned out by the negative class).
- Differential learning rates: backbone gets a much lower LR than the newly-initialized
  attention + classifier head, with a short linear warmup into cosine annealing.
- Series are now ordered using SeriesDescription (sagittal / coronal / axial) when
  available, so the model reliably sees complementary views instead of two of the same.
- More slices per series, more epochs, gradient clipping, early stopping per fold.
- Test-time augmentation (horizontal flip averaging) + AUC-weighted fold ensembling at
  inference, instead of a plain unweighted average.
"""

import os
import glob
import json
import math
import time
import random
import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

try:
    import pydicom
except ImportError:
    print("Warning: pydicom not pre-installed. Fallback mode will be used if DICOM files are missing.")
    pydicom = None

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')



# 1. GLOBAL CONFIGURATION


class Config:
    KAGGLE_INPUT_DIR = "/kaggle/input/rsna-knee-abnormality-detection"
    KAGGLE_COMPETITIONS_DIR = "/kaggle/input/competitions/rsna-knee-abnormality-detection"

    if os.path.exists(KAGGLE_COMPETITIONS_DIR):
        INPUT_DIR = KAGGLE_COMPETITIONS_DIR
    else:
        INPUT_DIR = KAGGLE_INPUT_DIR

    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    TRAIN_SERIES_CSV = os.path.join(INPUT_DIR, "train_series.csv")
    TEST_SERIES_CSV = os.path.join(INPUT_DIR, "test_series.csv")
    TRAIN_SERIES_DIR = os.path.join(INPUT_DIR, "train_series")
    TEST_SERIES_DIR = os.path.join(INPUT_DIR, "test_series")

    OUTPUT_DIR = "./output"
    SUBMISSION_PATH = "submission.csv"

    # --- Offline pretrained-weight staging ------------------------------
    # Point this at an attached Kaggle Dataset holding the .pth files.
    # Create it once (with internet ON, outside the scored run) via:
    #   import torchvision.models as models
    #   models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    #   models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    #   # then upload the resulting files from ~/.cache/torch/hub/checkpoints/
    #   # as a new Kaggle Dataset, and set PRETRAINED_WEIGHTS_DIRS below to its path.
    PRETRAINED_WEIGHTS_DIRS = [
        "/kaggle/input/pytorch-effnet-weights",
        "/kaggle/input/pytorch-pretrained-image-models",
        "/kaggle/input/pretrained-backbones",
        "/kaggle/input/torchvision-pretrained-weights",
    ]
    TARGET_COLS = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture"
    ]
    NUM_CLASSES = len(TARGET_COLS)

    IMAGE_SIZE = (224, 224)
    NUM_SLICES_PER_SERIES = 8          # cover more of each MRI acquisition
    MAX_SERIES_PER_STUDY = 3            # prefer sagittal, coronal, and axial views
    BACKBONE_NAME = "efficientnet_b0"
    PRETRAINED = True

    SEED = 42
    N_FOLDS = 5
    EPOCHS = 20                        # early stopping still limits wasted epochs
    WARMUP_EPOCHS = 1
    PATIENCE = 5                       # early stopping patience (epochs w/o AUC improvement)
    BATCH_SIZE = 8
    BACKBONE_LR = 1e-4                 # differential LR: backbone learns slowly
    HEAD_LR = 1e-3                     # head (new layers) learns faster
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP_NORM = 5.0
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MIXED_PRECISION = True

    # Kaggle allows roughly 9 hours per submission notebook. Keep a reserve
    # for final inference and CSV creation instead of risking a hard timeout.
    MAX_RUNTIME_SECONDS = 8 * 60 * 60
    INFERENCE_RESERVE_SECONDS = 45 * 60
    RUN_START_TIME = None

    # Data-quality choice: treat missing labels as "unknown" (masked out of loss)
    # rather than silently coercing them to 0.0 (confirmed negative). Set to False
    # to restore the old (riskier) behavior.
    MASK_MISSING_LABELS = True

    TTA_HFLIP = True


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



# 1b. OFFLINE PRETRAINED WEIGHT STAGING  (the critical fix)


def stage_offline_pretrained_weights(config: "Config") -> Optional[str]:
    """
    Finds a local torchvision checkpoint. The model loads this file directly,
    so no URL lookup is performed during an internet-disabled Kaggle run.

    Returns the checkpoint path or None when it is unavailable.
    """
    cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    os.makedirs(cache_dir, exist_ok=True)

    if config.BACKBONE_NAME == "efficientnet_b0":
        expected_name = os.path.basename(models.EfficientNet_B0_Weights.DEFAULT.url)
    elif config.BACKBONE_NAME == "resnet34":
        expected_name = os.path.basename(models.ResNet34_Weights.DEFAULT.url)
    else:
        print(f"[WeightStaging] Unsupported backbone '{config.BACKBONE_NAME}'.")
        return None

    expected_path = os.path.join(cache_dir, expected_name)
    if os.path.exists(expected_path):
        print(f"[WeightStaging] Using cached pretrained weights: {expected_path}")
        return expected_path

    search_roots = list(config.PRETRAINED_WEIGHTS_DIRS) + ["/kaggle/input", "."]
    for root in search_roots:
        if not os.path.exists(root):
            continue
        candidates = glob.glob(os.path.join(root, "**", expected_name), recursive=True)
        if not candidates:
            prefix = expected_name.split("-")[0]
            candidates = glob.glob(os.path.join(root, "**", f"{prefix}-*.pth"), recursive=True)
        if candidates:
            print(f"[WeightStaging] Found pretrained weights: {candidates[0]}")
            return candidates[0]

    print("[WeightStaging] ERROR: No exact offline pretrained weight file found in "
          f"{search_roots}. If this run has internet disabled (e.g. the scored "
          "submission), attach a Kaggle Dataset containing "
          f"'{expected_name}' and add its path to Config.PRETRAINED_WEIGHTS_DIRS.")
    return None


def load_local_backbone_weights(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")
    checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)
    print(f"[WeightStaging] Loaded verified local weights: {checkpoint_path}")



# 2. DICOM LOADING & 2.5D PROCESSING


def load_dicom_slice(path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    if pydicom is not None and os.path.exists(path):
        try:
            dcm = pydicom.dcmread(path, stop_before_pixels=False)
            img = dcm.pixel_array.astype(np.float32)

            slope = float(getattr(dcm, 'RescaleSlope', 1.0))
            intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
            img = img * slope + intercept

            p1, p99 = np.percentile(img, (1, 99))
            if p99 > p1:
                img = np.clip(img, p1, p99)
                img = (img - p1) / (p99 - p1) * 255.0
            else:
                img = np.zeros_like(img)

            img_uint8 = img.astype(np.uint8)
        except Exception:
            img_uint8 = np.zeros(target_size, dtype=np.uint8)
    else:
        img_uint8 = np.zeros(target_size, dtype=np.uint8)

    tensor_img = torch.from_numpy(img_uint8).unsqueeze(0).unsqueeze(0).float()
    tensor_resized = F.interpolate(tensor_img, size=target_size, mode='bilinear', align_corners=False)
    return tensor_resized.squeeze().numpy().astype(np.uint8)


def build_25d_triplets(slice_paths: List[str], n_samples: int, target_size: Tuple[int, int]) -> torch.Tensor:
    num_slices = len(slice_paths)
    if num_slices == 0:
        return torch.zeros((n_samples, 3, target_size[0], target_size[1]), dtype=torch.float32)

    if num_slices >= n_samples:
        indices = np.linspace(0, num_slices - 1, n_samples, dtype=int)
    else:
        indices = np.pad(np.arange(num_slices), (0, n_samples - num_slices), mode='edge')

    sampled_tensors = []
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    for idx in indices:
        prev_idx = max(0, idx - 1)
        curr_idx = idx
        next_idx = min(num_slices - 1, idx + 1)

        img_prev = load_dicom_slice(slice_paths[prev_idx], target_size)
        img_curr = load_dicom_slice(slice_paths[curr_idx], target_size)
        img_next = load_dicom_slice(slice_paths[next_idx], target_size)

        triplet = np.stack([img_prev, img_curr, img_next], axis=0).astype(np.float32) / 255.0
        triplet = (triplet - mean) / std

        sampled_tensors.append(torch.from_numpy(triplet).float())

    return torch.stack(sampled_tensors, dim=0)  # (N, 3, H, W)


def augment_series_tensor(series_tensor: torch.Tensor) -> torch.Tensor:
    """
    Lightweight, dependency-free augmentation applied identically across all
    N slices of a series so spatial correspondence between slices is preserved.
    series_tensor: (N, 3, H, W), already normalized.
    """
    # Random horizontal flip (~50%)
    if random.random() < 0.5:
        series_tensor = torch.flip(series_tensor, dims=[-1])

    # Random small rotation (-10 to +10 degrees), applied to whole stack at once
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        theta = math.radians(angle)
        cos_a, sin_a = math.cos(theta), math.sin(theta)
        N = series_tensor.shape[0]
        rot_mat = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=torch.float32)
        rot_mat = rot_mat.unsqueeze(0).repeat(N, 1, 1)
        grid = F.affine_grid(rot_mat, series_tensor.size(), align_corners=False)
        series_tensor = F.grid_sample(series_tensor, grid, align_corners=False, padding_mode="border")

    # Mild intensity jitter (helps generalize across scanner/protocol differences)
    if random.random() < 0.5:
        gain = random.uniform(0.9, 1.1)
        bias = random.uniform(-0.05, 0.05)
        series_tensor = series_tensor * gain + bias

    return series_tensor



# 3. PYTORCH DATASET


def _series_sort_key(series_id: str, series_df: Optional[pd.DataFrame]) -> Tuple[int, int, int]:
    """
    Orders series so complementary anatomical planes are preferred over picking
    two series of the same plane. Sagittal first (usually most informative for
    ACL/meniscus), then coronal, then axial, then everything else.
    """
    if series_df is None or "SeriesInstanceUID" not in series_df.columns:
        return (3, 0, 0)
    rows = series_df[series_df["SeriesInstanceUID"] == series_id]
    if rows.empty:
        return (3, 0, 0)
    row = rows.iloc[0]
    plane = str(row.get("Anatomical_Plane", "")).lower()
    plane_rank = {"sagittal": 0, "coronal": 1, "axial": 2}.get(plane, 3)
    fluid_rank = -int(row.get("Fluid_Sensitive", 0) == 1)
    fat_rank = -int(row.get("Fat_Suppression", 0) == 1)
    return (plane_rank, fluid_rank, fat_rank)


def sort_dicom_files(paths: List[str]) -> List[str]:
    """Sort slices by DICOM position/instance, never by opaque SOP UID."""
    def key(path: str):
        try:
            dcm = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=["InstanceNumber", "ImagePositionPatient"],
            )
            position = getattr(dcm, "ImagePositionPatient", None)
            position_key = tuple(float(value) for value in position) if position is not None else (float("inf"),)
            return (position_key, int(getattr(dcm, "InstanceNumber", 0)))
        except Exception:
            return ((float("inf"),), 0)
    return sorted(paths, key=key)


def select_series_ids(series_ids: List[str], series_df: Optional[pd.DataFrame], max_series: int) -> List[str]:
    """Select the strongest complementary planes before filling remaining slots."""
    if series_df is None or series_df.empty:
        return sorted(series_ids)[:max_series]

    metadata = series_df.set_index("SeriesInstanceUID", drop=False)
    candidates = [series_id for series_id in series_ids if series_id in metadata.index]
    selected = []
    for plane in ("sagittal", "coronal", "axial"):
        plane_candidates = [series_id for series_id in candidates
                            if str(metadata.loc[series_id].get("Anatomical_Plane", "")).lower() == plane]
        plane_candidates.sort(key=lambda series_id: (
            -int(metadata.loc[series_id].get("Fluid_Sensitive", 0) == 1),
            -int(metadata.loc[series_id].get("Fat_Suppression", 0) == 1),
        ))
        if plane_candidates:
            selected.append(plane_candidates[0])

    remaining = [series_id for series_id in series_ids if series_id not in selected]
    remaining.sort(key=lambda series_id: _series_sort_key(series_id, series_df))
    selected.extend(remaining)
    return selected[:max_series]


class RSNAKneeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, series_df: Optional[pd.DataFrame], base_dir: str,
                 config: "Config", is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.series_df = series_df
        self.base_dir = base_dir
        self.config = config
        self.is_train = is_train

        self.study_to_series = {}
        if series_df is not None and not series_df.empty and 'StudyInstanceUID' in series_df.columns:
            grouped = series_df.groupby('StudyInstanceUID')
            for study_id, group in grouped:
                self.study_to_series[study_id] = group['SeriesInstanceUID'].tolist()

        self.series_cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        study_id = row['StudyInstanceUID']

        labels = torch.zeros(self.config.NUM_CLASSES, dtype=torch.float32)
        label_mask = torch.ones(self.config.NUM_CLASSES, dtype=torch.float32)

        if all(c in row for c in self.config.TARGET_COLS):
            raw = row[self.config.TARGET_COLS]
            if self.config.MASK_MISSING_LABELS:
                label_mask = torch.tensor((~raw.isna()).values.astype(np.float32))
            val_array = raw.fillna(0.0).values.astype(np.float32)
            labels = torch.tensor(val_array)

        series_ids = self.study_to_series.get(study_id, [])
        study_path = os.path.join(self.base_dir, study_id)
        if not series_ids and os.path.exists(study_path):
            series_ids = [d for d in os.listdir(study_path) if os.path.isdir(os.path.join(study_path, d))]

        series_ids = select_series_ids(series_ids, self.series_df, self.config.MAX_SERIES_PER_STUDY)

        study_tensors = []
        for s_id in series_ids:
            series_dir = os.path.join(self.base_dir, study_id, s_id)
            if series_dir in self.series_cache:
                dcm_files = self.series_cache[series_dir]
            elif os.path.exists(series_dir):
                dcm_files = sort_dicom_files([os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith(".dcm")])
                self.series_cache[series_dir] = dcm_files
            else:
                dcm_files = []

            series_tensor = build_25d_triplets(dcm_files, self.config.NUM_SLICES_PER_SERIES, self.config.IMAGE_SIZE)
            if self.is_train:
                series_tensor = augment_series_tensor(series_tensor)
            study_tensors.append(series_tensor)

        while len(study_tensors) < self.config.MAX_SERIES_PER_STUDY:
            empty_series = torch.zeros(
                (self.config.NUM_SLICES_PER_SERIES, 3, self.config.IMAGE_SIZE[0], self.config.IMAGE_SIZE[1]),
                dtype=torch.float32
            )
            study_tensors.append(empty_series)

        study_tensor = torch.stack(study_tensors, dim=0)  # (S, N, 3, H, W)
        return study_tensor, labels, label_mask, study_id



# 4. NEURAL NETWORK ARCHITECTURE


class GatedAttentionPooling(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.attn_w = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.attn_v = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Sigmoid(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.attn_w(x)
        v = self.attn_v(x)
        scores = w * v
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(x * weights, dim=1)
        return pooled, weights


class RSNAKneeModel(nn.Module):
    def __init__(self, config: "Config", backbone_name: str = "efficientnet_b0",
                 num_classes: int = 12, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        checkpoint_path = stage_offline_pretrained_weights(config) if pretrained else None

        if "resnet" in backbone_name:
            if pretrained and checkpoint_path is None:
                raise RuntimeError("Offline pretrained weights are required but were not found.")
            resnet = models.resnet34(weights=None)
            if pretrained:
                load_local_backbone_weights(resnet, checkpoint_path)
            self.feature_dim = resnet.fc.in_features
            resnet.fc = nn.Identity()
            self.backbone = resnet
        else:
            if pretrained and checkpoint_path is None:
                raise RuntimeError("Offline pretrained weights are required but were not found.")
            effnet = models.efficientnet_b0(weights=None)
            if pretrained:
                load_local_backbone_weights(effnet, checkpoint_path)
            self.feature_dim = effnet.classifier[1].in_features
            effnet.classifier = nn.Identity()
            self.backbone = effnet

        self.dropout = nn.Dropout(0.3)
        self.attention = GatedAttentionPooling(in_features=self.feature_dim, hidden_dim=256)

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, N, C, H, W = x.shape
        x_flat = x.view(B * S * N, C, H, W)

        features = self.backbone(x_flat)
        features = self.dropout(features)

        features_seq = features.view(B, S * N, self.feature_dim)
        pooled_features, _ = self.attention(features_seq)

        logits = self.classifier(pooled_features)
        return logits

    def param_groups(self, backbone_lr: float, head_lr: float, weight_decay: float):
        """Differential LR: backbone (pretrained) learns slowly, new head learns fast."""
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": list(self.attention.parameters()) + list(self.classifier.parameters()),
             "lr": head_lr, "weight_decay": weight_decay},
        ]



# 5. METRIC EVALUATION


def calculate_macro_auc(y_true: np.ndarray, y_pred: np.ndarray, target_cols: List[str]) -> Tuple[float, Dict[str, float]]:
    aucs = {}
    valid_aucs = []

    for i, col in enumerate(target_cols):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        mask = ~np.isnan(y_t)
        y_t_clean = y_t[mask]
        y_p_clean = y_p[mask]

        if len(y_t_clean) > 0 and len(np.unique(y_t_clean)) > 1:
            try:
                score = roc_auc_score(y_t_clean, y_p_clean)
                aucs[col] = float(score)
                valid_aucs.append(score)
            except ValueError:
                aucs[col] = 0.5
        else:
            aucs[col] = 0.5

    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.5
    return macro_auc, aucs


def compute_pos_weight(df: pd.DataFrame, target_cols: List[str], device: str, clip_max: float = 20.0) -> torch.Tensor:
    """pos_weight[i] = (#negatives / #positives) for class i, used by BCEWithLogitsLoss
    to counteract class imbalance (rare abnormalities getting drowned out)."""
    weights = []
    for col in target_cols:
        vals = df[col].dropna()
        pos = max(vals.sum(), 1.0)
        neg = max(len(vals) - vals.sum(), 1.0)
        w = min(neg / pos, clip_max)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32, device=device)



# 6. TRAINING & VALIDATION LOOPS


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_clip_norm):
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    for step, (images, targets, mask, _) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss_raw = criterion(logits, targets)
                loss = (loss_raw * mask).sum() / mask.sum().clamp(min=1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss_raw = criterion(logits, targets)
            loss = (loss_raw * mask).sum() / mask.sum().clamp(min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 100 == 0 or (step + 1) == num_batches:
            avg = total_loss / (step + 1)
            pct = (step + 1) / num_batches * 100
            print(f"   Batch [{step+1:04d}/{num_batches:04d}] ({pct:5.1f}%) - Current Loss: {avg:.4f}", flush=True)

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, criterion, device, target_cols):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []

    for images, targets, mask, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        logits = model(images)
        loss_raw = criterion(logits, targets)
        loss = (loss_raw * mask).sum() / mask.sum().clamp(min=1.0)
        probs = torch.sigmoid(logits)

        total_loss += loss.item()
        # Respect the mask: mark unlabeled entries as NaN so they're excluded from AUC
        t = targets.cpu().numpy().copy()
        m = mask.cpu().numpy()
        t[m == 0] = np.nan
        all_targets.append(t)
        all_preds.append(probs.cpu().numpy())

    all_targets = np.vstack(all_targets) if all_targets else np.zeros((0, len(target_cols)))
    all_preds = np.vstack(all_preds) if all_preds else np.zeros((0, len(target_cols)))

    macro_auc, class_aucs = calculate_macro_auc(all_targets, all_preds, target_cols)
    avg_loss = total_loss / max(len(loader), 1)

    return avg_loss, macro_auc, class_aucs


def make_scheduler(optimizer, warmup_epochs, total_epochs):
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(warmup_epochs, 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=1e-6)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])



# 7. CROSS-VALIDATION PIPELINE


def run_cross_validation(train_df: pd.DataFrame, series_df: Optional[pd.DataFrame], config: "Config"):
    labeled_df = train_df.reset_index(drop=True)
    start_time = time.monotonic()
    training_deadline = config.MAX_RUNTIME_SECONDS - config.INFERENCE_RESERVE_SECONDS

    def training_time_expired() -> bool:
        return time.monotonic() - start_time >= training_deadline

    print("=" * 60)
    print("RSNA Knee Abnormality Detection: Cross-Validation (v2)")
    print(f"Total studies: {len(labeled_df)} | Folds: {config.N_FOLDS} | Epochs: {config.EPOCHS}")
    print("=" * 60)

    if len(labeled_df) == 0:
        raise ValueError("No labeled studies found! Check config.TARGET_COLS or your train.csv.")

    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)

    oof_predictions = np.zeros((len(labeled_df), config.NUM_CLASSES))
    oof_targets_masked = labeled_df[config.TARGET_COLS].values.astype(np.float32)  # keep NaNs for honest OOF AUC
    fold_aucs = {}

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(labeled_df)):
        if training_time_expired():
            print("Training time budget reached; starting inference with completed folds.")
            break
        print(f"\n--- Fold {fold + 1}/{config.N_FOLDS} (Train: {len(train_idx)}, Val: {len(val_idx)}) ---")

        fold_train_df = labeled_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = labeled_df.iloc[val_idx].reset_index(drop=True)

        train_ds = RSNAKneeDataset(fold_train_df, series_df, config.TRAIN_SERIES_DIR, config, is_train=True)
        val_ds = RSNAKneeDataset(fold_val_df, series_df, config.TRAIN_SERIES_DIR, config, is_train=False)

        train_loader = DataLoader(
            train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=(len(train_ds) > config.BATCH_SIZE)
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=config.NUM_WORKERS, pin_memory=True
        )

        model = RSNAKneeModel(config, backbone_name=config.BACKBONE_NAME,
                               num_classes=config.NUM_CLASSES, pretrained=config.PRETRAINED)
        model = model.to(config.DEVICE)

        pos_weight = compute_pos_weight(fold_train_df, config.TARGET_COLS, config.DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

        optimizer = torch.optim.AdamW(
            model.param_groups(config.BACKBONE_LR, config.HEAD_LR, config.WEIGHT_DECAY)
        )
        scheduler = make_scheduler(optimizer, config.WARMUP_EPOCHS, config.EPOCHS)
        scaler = torch.cuda.amp.GradScaler() if (config.MIXED_PRECISION and config.DEVICE == "cuda") else None

        best_val_auc = 0.0
        epochs_no_improve = 0
        model_save_path = os.path.join(config.OUTPUT_DIR, f"model_fold_{fold}.pth")

        for epoch in range(config.EPOCHS):
            if training_time_expired():
                print("Training time budget reached; ending at the last completed epoch.")
                break
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                                          config.DEVICE, config.GRAD_CLIP_NORM)
            val_loss, val_auc, _ = validate(model, val_loader, criterion, config.DEVICE, config.TARGET_COLS)
            scheduler.step()
            elapsed = time.time() - t0

            print(f"Epoch {epoch+1:02d}/{config.EPOCHS:02d} [{elapsed:.1f}s] - "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro AUC: {val_auc:.4f}", flush=True)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_save_path)
                print(f"  -> Saved Checkpoint (Best AUC: {best_val_auc:.4f})", flush=True)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.PATIENCE:
                    print(f"  -> Early stopping (no improvement in {config.PATIENCE} epochs)", flush=True)
                    break

        fold_aucs[fold] = best_val_auc

        if not os.path.exists(model_save_path):
            print(f"No completed checkpoint for fold {fold}; skipping this fold.")
            continue
        model.load_state_dict(torch.load(model_save_path, map_location=config.DEVICE))
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for images, _, _, _ in val_loader:
                images = images.to(config.DEVICE)
                logits = model(images)
                probs = torch.sigmoid(logits)
                if config.TTA_HFLIP:
                    logits_f = model(torch.flip(images, dims=[-1]))
                    probs = (probs + torch.sigmoid(logits_f)) / 2.0
                fold_preds.append(probs.cpu().numpy())

        if fold_preds:
            oof_predictions[val_idx] = np.vstack(fold_preds)

    total_oof_auc, class_aucs = calculate_macro_auc(oof_targets_masked, oof_predictions, config.TARGET_COLS)
    print("\n" + "=" * 60)
    print(f"FINAL OVERALL OOF MACRO AUC ROC: {total_oof_auc:.4f}")
    print("=" * 60)
    for col, score in class_aucs.items():
        print(f"  - {col:18s}: {score:.4f}")
    print("=" * 60)

    with open(os.path.join(config.OUTPUT_DIR, "fold_aucs.json"), "w") as f:
        json.dump(fold_aucs, f, indent=2)



# 8. INFERENCE & KAGGLE SUBMISSION ENGINE


def generate_submission(config: "Config"):
    print("\n" + "=" * 60)
    print("Generating Final Test Submission")
    print("=" * 60)

    if not os.path.exists(config.TEST_CSV):
        print(f"Test CSV path '{config.TEST_CSV}' not found. Generating sample fallback submission.")
        create_dummy_submission(config)
        return

    test_df = pd.read_csv(config.TEST_CSV)
    test_series_df = pd.read_csv(config.TEST_SERIES_CSV) if os.path.exists(config.TEST_SERIES_CSV) else None

    test_ds = RSNAKneeDataset(test_df, test_series_df, config.TEST_SERIES_DIR, config, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    model_paths = sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "model_fold_*.pth")))

    if not model_paths:
        print("No trained checkpoints found! Generating default baseline predictions (0.5).")
        for col in config.TARGET_COLS:
            test_df[col] = 0.5
        test_df[['StudyInstanceUID'] + config.TARGET_COLS].to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Saved fallback to {config.SUBMISSION_PATH}")
        return

    # Load per-fold AUCs (if available) to weight the ensemble by fold quality
    fold_aucs_path = os.path.join(config.OUTPUT_DIR, "fold_aucs.json")
    fold_aucs = {}
    if os.path.exists(fold_aucs_path):
        with open(fold_aucs_path) as f:
            fold_aucs = {int(k): v for k, v in json.load(f).items()}

    print(f"Found {len(model_paths)} model checkpoint(s) for ensemble inference.")
    all_model_preds = np.zeros((len(test_df), config.NUM_CLASSES), dtype=np.float32)
    total_weight = 0.0

    for p in model_paths:
        if config.RUN_START_TIME is not None:
            elapsed = time.monotonic() - config.RUN_START_TIME
            if elapsed >= config.MAX_RUNTIME_SECONDS:
                print("Runtime budget reached during inference; using completed models only.")
                break
        fold_idx = int(os.path.basename(p).split("_")[-1].split(".")[0])
        weight = fold_aucs.get(fold_idx, 0.5)  # default equal weight if unknown
        weight = max(weight, 0.05)  # avoid zeroing out a fold entirely

        print(f"Running inference with model: {os.path.basename(p)} (ensemble weight={weight:.4f})")
        model = RSNAKneeModel(config, backbone_name=config.BACKBONE_NAME,
                               num_classes=config.NUM_CLASSES, pretrained=False)
        model.load_state_dict(torch.load(p, map_location=config.DEVICE))
        model.to(config.DEVICE)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for images, _, _, _ in test_loader:
                images = images.to(config.DEVICE)
                logits = model(images)
                probs = torch.sigmoid(logits)
                if config.TTA_HFLIP:
                    logits_f = model(torch.flip(images, dims=[-1]))
                    probs = (probs + torch.sigmoid(logits_f)) / 2.0
                fold_preds.append(probs.cpu().numpy())

        fold_preds_arr = np.vstack(fold_preds)
        all_model_preds += fold_preds_arr * weight
        total_weight += weight

    all_model_preds /= max(total_weight, 1e-8)

    sub_df = pd.DataFrame({'StudyInstanceUID': test_df['StudyInstanceUID']})
    for i, col in enumerate(config.TARGET_COLS):
        sub_df[col] = np.clip(all_model_preds[:, i], 0.0, 1.0)

    sub_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Successfully generated Kaggle submission: '{config.SUBMISSION_PATH}'")
    print(f"Submission Shape: {sub_df.shape}")
    print("First 3 rows:")
    print(sub_df.head(3))


def create_dummy_submission(config: "Config"):
    dummy_data = {'StudyInstanceUID': ['test_study_001', 'test_study_002', 'test_study_003']}
    for col in config.TARGET_COLS:
        dummy_data[col] = [0.5, 0.5, 0.5]
    sub = pd.DataFrame(dummy_data)
    sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Created fallback dummy submission: {config.SUBMISSION_PATH}")



# 9. MAIN EXECUTION


if __name__ == "__main__":
    Config.RUN_START_TIME = time.monotonic()
    seed_everything(Config.SEED)

    # Stage pretrained weights ONCE up front, before any model is built, so both
    # training and inference paths see a consistent, network-free weight source.
    stage_offline_pretrained_weights(Config)

    if os.path.exists(Config.TRAIN_CSV):
        print("Kaggle training data detected. Loading datasets...")
        train_df = pd.read_csv(Config.TRAIN_CSV)
        series_df = pd.read_csv(Config.TRAIN_SERIES_CSV) if os.path.exists(Config.TRAIN_SERIES_CSV) else None

        run_cross_validation(train_df, series_df, Config)
        generate_submission(Config)
    else:
        print("Dataset not found at Kaggle path. Generating fallback submission.")
        generate_submission(Config)
