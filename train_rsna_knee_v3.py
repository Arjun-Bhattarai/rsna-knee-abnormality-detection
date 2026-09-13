import os
import glob
import json
import random
import time
import warnings

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
    EPOCHS = 10
    PATIENCE = 3
    WARMUP_EPOCHS = 1
    BACKBONE_LR = 3e-5
    HEAD_LR = 8e-4
    WEIGHT_DECAY = 2e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    AMP = DEVICE == "cuda"
    PRETRAINED = True
    MASK_MISSING = True
    MAX_RUNTIME = 9 * 60 * 60
    INFERENCE_RESERVE = 25 * 60
    CACHE_DTYPE = torch.float16 if AMP else torch.float32


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


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
            orientation = getattr(dcm, "ImageOrientationPatient", None)
            if position is not None and orientation is not None:
                row = np.asarray(orientation[:3], dtype=np.float64)
                column = np.asarray(orientation[3:], dtype=np.float64)
                normal = np.cross(row, column)
                position = float(np.dot(np.asarray(position, dtype=np.float64), normal))
            else:
                position = float("inf")
            return position, int(getattr(dcm, "InstanceNumber", 0))
        except Exception:
            return float("inf"), 0

    return sorted(paths, key=key)


def make_stack(paths):
    if not paths:
        return torch.zeros(
            Config.NUM_SLICES, 3, *Config.IMAGE_SIZE, dtype=Config.CACHE_DTYPE
        )
    indices = np.linspace(0, len(paths) - 1, Config.NUM_SLICES, dtype=int)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    decoded = {}
    for index in set(indices.tolist()):
        decoded[index] = load_dicom(paths[index])
    output = []
    for index in indices:
        before_index = max(0, index - 1)
        after_index = min(len(paths) - 1, index + 1)
        for neighbor in (before_index, after_index):
            if neighbor not in decoded:
                decoded[neighbor] = load_dicom(paths[neighbor])
        image = torch.from_numpy(
            np.stack([decoded[before_index], decoded[index], decoded[after_index]])
        ).float()
        output.append((image - mean) / std)
    return torch.stack(output).to(dtype=Config.CACHE_DTYPE)


def augment(x):
    if random.random() < 0.5:
        x = torch.flip(x, [-1])
    if random.random() < 0.25:
        scale = random.uniform(0.92, 1.08)
        shift = random.uniform(-0.05, 0.05)
        x = x * scale + shift
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


def build_series_index(series_df):
    if series_df is None or series_df.empty:
        return {}
    return {
        study_id: group.SeriesInstanceUID.tolist()
        for study_id, group in series_df.groupby("StudyInstanceUID")
    }


def preload_stacks(study_ids, study_series, series_df, base_dir):
    cache = {}
    total = len(study_ids)
    for number, study_id in enumerate(study_ids, 1):
        for series_id in choose_series(study_series.get(study_id, []), series_df):
            directory = os.path.join(base_dir, study_id, series_id)
            if os.path.isdir(directory):
                files = [
                    os.path.join(directory, item)
                    for item in os.listdir(directory)
                    if item.lower().endswith(".dcm")
                ]
                cache[(study_id, series_id)] = make_stack(sort_dicom_files(files))
            else:
                cache[(study_id, series_id)] = make_stack([])
        if number % 25 == 0 or number == total:
            print(f"Preloaded {number}/{total} studies", flush=True)
    return cache


class KneeDataset(Dataset):
    def __init__(self, dataframe, series_df, base_dir, training, stack_cache=None, study_series=None):
        self.df = dataframe.reset_index(drop=True)
        self.series_df = series_df
        self.base_dir = base_dir
        self.training = training
        self.study_series = study_series if study_series is not None else build_series_index(series_df)
        self.stack_cache = stack_cache if stack_cache is not None else {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        study_id = row.StudyInstanceUID
        labels = torch.zeros(len(Config.TARGETS), dtype=torch.float32)
        mask = torch.ones(len(Config.TARGETS), dtype=torch.float32)
        values = pd.Series(index=Config.TARGETS, dtype=np.float32)
        for column in Config.TARGETS:
            if column in row.index:
                values[column] = row[column]
        if Config.MASK_MISSING:
            mask = torch.tensor((~values.isna()).to_numpy(dtype=np.float32))
        labels = torch.tensor(values.fillna(0).to_numpy(dtype=np.float32))

        series_ids = choose_series(self.study_series.get(study_id, []), self.series_df)
        stacks = []
        for series_id in series_ids:
            stack = self.stack_cache.get((study_id, series_id))
            if stack is None:
                stack = make_stack([])
            stacks.append(augment(stack) if self.training else stack)
        while len(stacks) < Config.MAX_SERIES:
            stacks.append(torch.zeros(
                Config.NUM_SLICES, 3, *Config.IMAGE_SIZE, dtype=Config.CACHE_DTYPE
            ))
        return torch.stack(stacks[:Config.MAX_SERIES]), labels, mask, study_id


class Attention(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(features, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, x, mask=None):
        scores = self.score(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        if mask is not None:
            weights = weights * mask.unsqueeze(-1).float()
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
        self.slice_attention = Attention(features)
        self.view_attention = Attention(features)
        fusion_dim = features * 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.20),
        )
        self.head = nn.Linear(256, len(Config.TARGETS))
        self.features = features

    def forward(self, x):
        batch, series, slices, channels, height, width = x.shape
        view_present = x.abs().sum(dim=(2, 3, 4, 5)) > 0
        encoded = self.backbone(
            x.reshape(batch * series * slices, channels, height, width)
        )
        encoded = encoded.reshape(batch, series, slices, self.features)
        view_features = []
        for view_index in range(series):
            slice_mask = view_present[:, view_index].unsqueeze(1).expand(-1, slices)
            pooled = self.slice_attention(encoded[:, view_index], slice_mask)
            pooled = pooled * view_present[:, view_index].unsqueeze(-1).float()
            view_features.append(pooled)
        views = torch.stack(view_features, dim=1)
        global_feature = self.view_attention(views, view_present)
        fused = torch.cat([views[:, 0], views[:, 1], views[:, 2], global_feature], dim=1)
        return self.head(self.fusion(fused))


def pos_weights(frame):
    values = []
    for column in Config.TARGETS:
        series = frame[column].dropna()
        positives = max(float(series.sum()), 1.0)
        negatives = max(float(len(series) - series.sum()), 1.0)
        values.append(min(negatives / positives, 20.0))
    return torch.tensor(values, dtype=torch.float32, device=Config.DEVICE)


def smooth_targets(labels, mask, smoothing=0.02):
    smoothed = labels * (1.0 - smoothing) + 0.5 * smoothing
    return torch.where(mask > 0, smoothed, labels)


def auc_score(targets, predictions):
    scores = []
    for index in range(len(Config.TARGETS)):
        valid = ~np.isnan(targets[:, index])
        if valid.sum() and len(np.unique(targets[valid, index])) > 1:
            scores.append(roc_auc_score(targets[valid, index], predictions[valid, index]))
    return float(np.mean(scores)) if scores else 0.5


def predict(model, images, tta=False):
    if Config.AMP:
        with torch.amp.autocast("cuda"):
            original = torch.sigmoid(model(images))
            flipped = torch.sigmoid(model(torch.flip(images, dims=[-1]))) if tta else None
    else:
        original = torch.sigmoid(model(images))
        flipped = torch.sigmoid(model(torch.flip(images, dims=[-1]))) if tta else None
    if flipped is None:
        return original.float().cpu().numpy()
    return ((original + flipped) * 0.5).float().cpu().numpy()


def loaders(train_frame, valid_frame, series_df, stack_cache, study_series):
    train = KneeDataset(
        train_frame, series_df, Config.TRAIN_SERIES_DIR, True,
        stack_cache=stack_cache, study_series=study_series
    )
    valid = KneeDataset(
        valid_frame, series_df, Config.TRAIN_SERIES_DIR, False,
        stack_cache=stack_cache, study_series=study_series
    )
    kwargs = dict(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.DEVICE == "cuda",
        persistent_workers=Config.NUM_WORKERS > 0,
    )
    if Config.NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = 4
    return (DataLoader(train, shuffle=True, drop_last=True, **kwargs),
            DataLoader(valid, shuffle=False, **kwargs))


def train_fold(train_frame, valid_frame, series_df, fold, deadline, stack_cache, study_series):
    train_loader, valid_loader = loaders(
        train_frame, valid_frame, series_df, stack_cache, study_series
    )
    model = KneeModel(True).to(Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights(train_frame), reduction="none")
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": Config.BACKBONE_LR},
        {
            "params": list(model.slice_attention.parameters())
                      + list(model.view_attention.parameters())
                      + list(model.fusion.parameters())
                      + list(model.head.parameters()),
            "lr": Config.HEAD_LR
        },
    ], weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(Config.EPOCHS - Config.WARMUP_EPOCHS, 1), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=Config.AMP) if Config.AMP else None
    ema = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if torch.is_floating_point(value)
    }
    ema_decay = 0.995
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
            labels_train = smooth_targets(labels, mask)
            if Config.AMP:
                with torch.amp.autocast("cuda"):
                    loss_values = criterion(model(images), labels_train)
                    loss = (loss_values * mask).sum() / mask.sum().clamp_min(1.0)
            else:
                loss_values = criterion(model(images), labels_train)
                loss = (loss_values * mask).sum() / mask.sum().clamp_min(1.0)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            with torch.no_grad():
                state = model.state_dict()
                for key in ema:
                    ema[key].mul_(ema_decay).add_(state[key].detach(), alpha=1.0 - ema_decay)

        model.eval()
        backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }
        current_state = model.state_dict()
        for key in ema:
            current_state[key].copy_(ema[key])
        targets, predictions = [], []
        with torch.no_grad():
            for images, labels, mask, _ in valid_loader:
                predictions.append(predict(
                    model, images.to(Config.DEVICE, non_blocking=True), tta=False))
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
                for key in backup:
                    current_state[key].copy_(backup[key])
                break
        for key in backup:
            current_state[key].copy_(backup[key])
        if epoch + 1 >= Config.WARMUP_EPOCHS:
            scheduler.step()
    return best_auc


def train_all(train_frame, series_df):
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    for path in glob.glob(os.path.join(Config.OUTPUT_DIR, "model_fold_*.pth")):
        os.remove(path)
    deadline = time.monotonic() + Config.MAX_RUNTIME - Config.INFERENCE_RESERVE
    study_series = build_series_index(series_df)
    study_ids = train_frame.StudyInstanceUID.drop_duplicates().tolist()
    print("Preloading DICOM stacks once...", flush=True)
    stack_cache = preload_stacks(
        study_ids, study_series, series_df, Config.TRAIN_SERIES_DIR
    )
    if time.monotonic() >= deadline:
        raise RuntimeError("Preprocessing used the available training budget.")
    splitter = KFold(Config.N_FOLDS, shuffle=True, random_state=42)
    fold_scores = {}
    for fold, (train_index, valid_index) in enumerate(splitter.split(train_frame)):
        if time.monotonic() >= deadline:
            print("Training budget reached; moving to inference.")
            break
        score = train_fold(
            train_frame.iloc[train_index], train_frame.iloc[valid_index],
            series_df, fold, deadline, stack_cache, study_series
        )
        fold_scores[fold] = score
    with open(os.path.join(Config.OUTPUT_DIR, "fold_aucs.json"), "w") as handle:
        json.dump(fold_scores, handle)


def generate_submission(test_frame, series_df):
    paths = sorted(glob.glob(os.path.join(Config.OUTPUT_DIR, "model_fold_*.pth")))
    if not paths:
        raise RuntimeError("No trained model checkpoints were produced.")
    study_series = build_series_index(series_df)
    study_ids = test_frame.StudyInstanceUID.drop_duplicates().tolist()
    print("Preloading test DICOM stacks once...", flush=True)
    stack_cache = preload_stacks(
        study_ids, study_series, series_df, Config.TEST_SERIES_DIR
    )
    dataset = KneeDataset(
        test_frame, series_df, Config.TEST_SERIES_DIR, False,
        stack_cache=stack_cache, study_series=study_series
    )
    loader_kwargs = dict(
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.DEVICE == "cuda",
        persistent_workers=Config.NUM_WORKERS > 0,
    )
    if Config.NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = 4
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
                output.append(predict(
                    model, images.to(Config.DEVICE, non_blocking=True), tta=True))
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
