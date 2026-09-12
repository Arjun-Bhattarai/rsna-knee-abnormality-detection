import os
import glob
import json
import math
import random
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


class Config:
    INPUT_DIR = "/kaggle/input/rsna-knee-abnormality-detection"
    ALT_INPUT_DIR = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
    if os.path.exists(ALT_INPUT_DIR):
        INPUT_DIR = ALT_INPUT_DIR

    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    TRAIN_SERIES_CSV = os.path.join(INPUT_DIR, "train_series.csv")
    TEST_SERIES_CSV = os.path.join(INPUT_DIR, "test_series.csv")
    TRAIN_SERIES_DIR = os.path.join(INPUT_DIR, "train_series")
    TEST_SERIES_DIR = os.path.join(INPUT_DIR, "test_series")

    OUTPUT_DIR = "./output"
    SUBMISSION_PATH = "submission.csv"
    WEIGHT_DIRS = [
        "/kaggle/input/pytorch-effnet-weights",
        "/kaggle/input/pytorch-pretrained-image-models",
        "/kaggle/input/pretrained-backbones",
        "/kaggle/input/torchvision-pretrained-weights",
    ]

    TARGETS = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture"
    ]
    IMAGE_SIZE = (224, 224)
    NUM_SLICES = 6
    MAX_SERIES = 3
    BATCH_SIZE = 8
    NUM_WORKERS = min(4, os.cpu_count() or 1)
    N_FOLDS = 3
    EPOCHS = 12
    PATIENCE = 4
    WARMUP_EPOCHS = 1
    BACKBONE_LR = 1e-4
    HEAD_LR = 1e-3
    WEIGHT_DECAY = 1e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    AMP = DEVICE == "cuda"
    PRETRAINED = True
    MASK_MISSING = True
    MAX_RUNTIME = 8 * 60 * 60
    INFERENCE_RESERVE = 20 * 60


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def pretrained_path():
    name = os.path.basename(models.EfficientNet_B0_Weights.DEFAULT.url)
    cache = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    os.makedirs(cache, exist_ok=True)
    cached = os.path.join(cache, name)
    if os.path.exists(cached):
        return cached
    roots = Config.WEIGHT_DIRS + ["/kaggle/input", "."]
    for root in roots:
        if os.path.exists(root):
            matches = glob.glob(os.path.join(root, "**", name), recursive=True)
            if matches:
                return matches[0]
    raise FileNotFoundError(
        f"Offline EfficientNet weights not found. Attach a Kaggle dataset containing {name}."
    )


def load_dicom(path):
    try:
        import pydicom
        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array.astype(np.float32)
        image = image * float(getattr(dcm, "RescaleSlope", 1.0))
        image += float(getattr(dcm, "RescaleIntercept", 0.0))
        low, high = np.percentile(image, (1, 99))
        image = np.clip(image, low, high)
        if high > low:
            image = (image - low) / (high - low)
        else:
            image = np.zeros_like(image)
        tensor = torch.from_numpy(image).float()[None, None]
        tensor = F.interpolate(tensor, size=Config.IMAGE_SIZE, mode="bilinear", align_corners=False)
        return tensor.squeeze().numpy()
    except Exception:
        return np.zeros(Config.IMAGE_SIZE, dtype=np.float32)


def sort_dicom_files(paths):
    try:
        import pydicom
    except ImportError:
        return sorted(paths)

    def key(path):
        try:
            dcm = pydicom.dcmread(path, stop_before_pixels=True)
            position = getattr(dcm, "ImagePositionPatient", None)
            position = tuple(float(x) for x in position) if position is not None else (float("inf"),)
            return position, int(getattr(dcm, "InstanceNumber", 0))
        except Exception:
            return (float("inf"),), 0

    return sorted(paths, key=key)


def make_stack(paths):
    if not paths:
        return torch.zeros(Config.NUM_SLICES, 3, *Config.IMAGE_SIZE)
    indices = np.linspace(0, len(paths) - 1, Config.NUM_SLICES, dtype=int)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    output = []
    for index in indices:
        before = load_dicom(paths[max(0, index - 1)])
        current = load_dicom(paths[index])
        after = load_dicom(paths[min(len(paths) - 1, index + 1)])
        image = np.stack([before, current, after], axis=0)
        output.append(torch.from_numpy((image - mean) / std).float())
    return torch.stack(output)


def augment(x):
    if random.random() < 0.5:
        x = torch.flip(x, [-1])
    if random.random() < 0.35:
        angle = math.radians(random.uniform(-8, 8))
        matrix = torch.tensor([[math.cos(angle), -math.sin(angle), 0],
                               [math.sin(angle), math.cos(angle), 0]], dtype=x.dtype)
        matrix = matrix.unsqueeze(0).repeat(x.shape[0], 1, 1)
        grid = F.affine_grid(matrix, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
    return x


def choose_series(series_ids, series_df):
    if not series_ids:
        return []
    if series_df is None or series_df.empty:
        return sorted(series_ids)[:Config.MAX_SERIES]
    metadata = series_df[series_df.SeriesInstanceUID.isin(series_ids)].copy()
    if metadata.empty:
        return sorted(series_ids)[:Config.MAX_SERIES]
    metadata = metadata.set_index("SeriesInstanceUID", drop=False)
    selected = []
    for plane in ("sagittal", "coronal", "axial"):
        candidates = [series_id for series_id in series_ids if series_id in metadata.index
                      and str(metadata.loc[series_id].get("Anatomical_Plane", "")).lower() == plane]
        candidates.sort(key=lambda series_id: (
            -int(metadata.loc[series_id].get("Fluid_Sensitive", 0) == 1),
            -int(metadata.loc[series_id].get("Fat_Suppression", 0) == 1),
        ))
        if candidates:
            selected.append(candidates[0])
            if len(selected) == Config.MAX_SERIES:
                return selected
    remaining = [series_id for series_id in series_ids if series_id not in selected]
    remaining.sort(key=lambda series_id: (
        -int(series_id in metadata.index and metadata.loc[series_id].get("Fluid_Sensitive", 0) == 1),
        -int(series_id in metadata.index and metadata.loc[series_id].get("Fat_Suppression", 0) == 1),
    ))
    for series_id in remaining:
        if series_id not in selected:
            selected.append(series_id)
        if len(selected) == Config.MAX_SERIES:
            break
    return selected


class KneeDataset(Dataset):
    def __init__(self, dataframe, series_df, base_dir, training):
        self.df = dataframe.reset_index(drop=True)
        self.series_df = series_df
        self.base_dir = base_dir
        self.training = training
        self.study_series = {}
        self.cache = {}
        if series_df is not None and "StudyInstanceUID" in series_df:
            for study_id, group in series_df.groupby("StudyInstanceUID"):
                self.study_series[study_id] = group.SeriesInstanceUID.tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        study_id = row.StudyInstanceUID
        labels = torch.zeros(len(Config.TARGETS), dtype=torch.float32)
        mask = torch.ones(len(Config.TARGETS), dtype=torch.float32)
        if all(column in row.index for column in Config.TARGETS):
            values = row[Config.TARGETS]
            if Config.MASK_MISSING:
                mask = torch.tensor((~values.isna()).to_numpy(dtype=np.float32))
            labels = torch.tensor(values.fillna(0).to_numpy(dtype=np.float32))

        series_ids = choose_series(self.study_series.get(study_id, []), self.series_df)
        stacks = []
        for series_id in series_ids:
            directory = os.path.join(self.base_dir, study_id, series_id)
            if directory not in self.cache:
                files = [os.path.join(directory, item) for item in os.listdir(directory)] if os.path.exists(directory) else []
                self.cache[directory] = sort_dicom_files([item for item in files if item.lower().endswith(".dcm")])
            stack = make_stack(self.cache[directory])
            stacks.append(augment(stack) if self.training else stack)
        while len(stacks) < Config.MAX_SERIES:
            stacks.append(torch.zeros(Config.NUM_SLICES, 3, *Config.IMAGE_SIZE))
        return torch.stack(stacks[:Config.MAX_SERIES]), labels, mask, study_id


class Attention(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(features, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return (x * weights).sum(1)


class KneeModel(nn.Module):
    def __init__(self, use_pretrained=True):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        if use_pretrained:
            checkpoint = torch.load(pretrained_path(), map_location="cpu")
            backbone.load_state_dict(checkpoint, strict=True)
        features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.attention = Attention(features)
        self.head = nn.Sequential(nn.Linear(features, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(.25), nn.Linear(256, 12))
        self.features = features

    def forward(self, x):
        batch, series, slices, channels, height, width = x.shape
        encoded = self.backbone(x.reshape(batch * series * slices, channels, height, width))
        encoded = encoded.reshape(batch, series * slices, self.features)
        return self.head(self.attention(encoded))


def pos_weights(frame):
    values = []
    for column in Config.TARGETS:
        series = frame[column].dropna()
        positives = max(float(series.sum()), 1.0)
        negatives = max(float(len(series) - series.sum()), 1.0)
        values.append(min(negatives / positives, 20.0))
    return torch.tensor(values, dtype=torch.float32, device=Config.DEVICE)


def auc_score(targets, predictions):
    scores = []
    for index in range(len(Config.TARGETS)):
        valid = ~np.isnan(targets[:, index])
        if valid.sum() and len(np.unique(targets[valid, index])) > 1:
            scores.append(roc_auc_score(targets[valid, index], predictions[valid, index]))
    return float(np.mean(scores)) if scores else 0.5


def predict_with_tta(model, images):
    original = torch.sigmoid(model(images))
    flipped = torch.sigmoid(model(torch.flip(images, dims=[-1])))
    return ((original + flipped) * 0.5).cpu().numpy()


def loaders(train_frame, valid_frame, series_df):
    train = KneeDataset(train_frame, series_df, Config.TRAIN_SERIES_DIR, True)
    valid = KneeDataset(valid_frame, series_df, Config.TRAIN_SERIES_DIR, False)
    kwargs = dict(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.DEVICE == "cuda",
        persistent_workers=Config.NUM_WORKERS > 0,
    )
    if Config.NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = 2
    return (DataLoader(train, shuffle=True, drop_last=True, **kwargs),
            DataLoader(valid, shuffle=False, **kwargs))


def train_fold(train_frame, valid_frame, series_df, fold, deadline):
    train_loader, valid_loader = loaders(train_frame, valid_frame, series_df)
    model = KneeModel(True).to(Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights(train_frame), reduction="none")
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": Config.BACKBONE_LR},
        {"params": list(model.attention.parameters()) + list(model.head.parameters()), "lr": Config.HEAD_LR},
    ], weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(Config.EPOCHS - Config.WARMUP_EPOCHS, 1), eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=Config.AMP)
    best_auc, stale = 0.0, 0
    path = os.path.join(Config.OUTPUT_DIR, f"model_fold_{fold}.pth")

    for epoch in range(Config.EPOCHS):
        if time.monotonic() >= deadline:
            break
        model.train()
        for images, labels, mask, _ in train_loader:
            images = images.to(Config.DEVICE, non_blocking=True)
            labels = labels.to(Config.DEVICE, non_blocking=True)
            mask = mask.to(Config.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=Config.AMP):
                loss_values = criterion(model(images), labels)
                loss = (loss_values * mask).sum() / mask.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        targets, predictions = [], []
        with torch.no_grad():
            for images, labels, mask, _ in valid_loader:
                predictions.append(predict_with_tta(
                    model, images.to(Config.DEVICE, non_blocking=True)))
                current = labels.numpy().copy()
                current[mask.numpy() == 0] = np.nan
                targets.append(current)
        score = auc_score(np.vstack(targets), np.vstack(predictions))
        print(f"Fold {fold + 1}/{Config.N_FOLDS}, epoch {epoch + 1}/{Config.EPOCHS}, AUC={score:.4f}", flush=True)
        if score > best_auc:
            best_auc, stale = score, 0
            torch.save(model.state_dict(), path)
        else:
            stale += 1
            if stale >= Config.PATIENCE:
                break
        if epoch + 1 >= Config.WARMUP_EPOCHS:
            scheduler.step()
    return best_auc


def train_all(train_frame, series_df):
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    for path in glob.glob(os.path.join(Config.OUTPUT_DIR, "model_fold_*.pth")):
        os.remove(path)
    deadline = time.monotonic() + Config.MAX_RUNTIME - Config.INFERENCE_RESERVE
    splitter = KFold(Config.N_FOLDS, shuffle=True, random_state=42)
    fold_scores = {}
    for fold, (train_index, valid_index) in enumerate(splitter.split(train_frame)):
        if time.monotonic() >= deadline:
            print("Training budget reached; moving to inference.")
            break
        score = train_fold(train_frame.iloc[train_index], train_frame.iloc[valid_index], series_df, fold, deadline)
        fold_scores[fold] = score
    with open(os.path.join(Config.OUTPUT_DIR, "fold_aucs.json"), "w") as handle:
        json.dump(fold_scores, handle)


def generate_submission(test_frame, series_df):
    paths = sorted(glob.glob(os.path.join(Config.OUTPUT_DIR, "model_fold_*.pth")))
    if not paths:
        raise RuntimeError("No trained model checkpoints were produced.")
    dataset = KneeDataset(test_frame, series_df, Config.TEST_SERIES_DIR, False)
    loader_kwargs = dict(
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.DEVICE == "cuda",
        persistent_workers=Config.NUM_WORKERS > 0,
    )
    if Config.NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)
    predictions = np.zeros((len(test_frame), len(Config.TARGETS)), dtype=np.float32)
    total_weight = 0.0
    with open(os.path.join(Config.OUTPUT_DIR, "fold_aucs.json")) as handle:
        scores = {int(k): float(v) for k, v in json.load(handle).items()}
    for path in paths:
        fold = int(os.path.basename(path).split("_")[-1].split(".")[0])
        model = KneeModel(False).to(Config.DEVICE)
        model.load_state_dict(torch.load(path, map_location=Config.DEVICE))
        model.eval()
        output = []
        with torch.no_grad():
            for images, _, _, _ in loader:
                output.append(predict_with_tta(
                    model, images.to(Config.DEVICE, non_blocking=True)))
        weight = max(scores.get(fold, 0.5), 0.05)
        predictions += np.vstack(output) * weight
        total_weight += weight
    predictions /= total_weight
    submission = pd.DataFrame({"StudyInstanceUID": test_frame.StudyInstanceUID})
    for index, column in enumerate(Config.TARGETS):
        submission[column] = np.clip(predictions[:, index], 0, 1)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Saved {Config.SUBMISSION_PATH}: {submission.shape}")


def main():
    seed_everything()
    if Config.DEVICE != "cuda":
        print("WARNING: CUDA is unavailable. Kaggle GPU acceleration is required for this runtime budget.")
    train = pd.read_csv(Config.TRAIN_CSV)
    test = pd.read_csv(Config.TEST_CSV)
    train_series = pd.read_csv(Config.TRAIN_SERIES_CSV) if os.path.exists(Config.TRAIN_SERIES_CSV) else None
    test_series = pd.read_csv(Config.TEST_SERIES_CSV) if os.path.exists(Config.TEST_SERIES_CSV) else None
    train_all(train, train_series)
    generate_submission(test, test_series)


if __name__ == "__main__":
    main()
