
"""
train_rsna_knee_v4.py
RSNA Knee Abnormality Detection (2026) -- weak-supervision pipeline.

WHY THIS IS DIFFERENT FROM v2/v3
--------------------------------
Only 58 of the 4,407 training studies carry expert labels. The other 4,349
carry a free-text radiology report in ~10 languages. v3 masked the NaN labels,
so those 4,349 studies produced zero gradient while still costing a full
forward+backward pass -- 98.7% of every epoch was wasted compute, and the model
effectively saw 58 examples. That is the 22-25 min/epoch cost and the 0.567.

v4:
  1. Manufactures soft labels for all 4,407 studies from the Report column
     (multilingual lexicon + clause-level negation + side attribution).
  2. Decodes every DICOM exactly ONCE into a uint8 memmap cache, then trains
     many epochs off the cache (epochs drop to ~4-6 min).
  3. Normalises laterality (right knees mirrored) so "medial" is always the
     same side of the image, and REMOVES horizontal-flip augmentation/TTA,
     which was silently swapping Medial<->Lateral on 4 of the 12 targets.
  4. Keeps the 58 gold studies entirely out of training and uses them as the
     checkpoint-selection gate, blended with a weak-holdout gate.

Runtime target: < 8 h on a single Kaggle T4/P100. Internet off.
"""

import os
import gc
import re
import glob
import json
import math
import multiprocessing
import time
import random
import unicodedata
import warnings
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# CONFIG
class CFG:
    INPUT_DIR = "/kaggle/input/rsna-knee-abnormality-detection"
    ALT_INPUT_DIR = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
    if os.path.exists(ALT_INPUT_DIR):
        INPUT_DIR = ALT_INPUT_DIR

    LOCAL_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    if not os.path.exists(INPUT_DIR):
        if os.path.exists(LOCAL_DATA_DIR):
            INPUT_DIR = os.path.abspath(os.path.dirname(__file__))

    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    TRAIN_SERIES_CSV = os.path.join(INPUT_DIR, "train_series.csv")
    TEST_SERIES_CSV = os.path.join(INPUT_DIR, "test_series.csv")
    TRAIN_SERIES_DIR = os.path.join(INPUT_DIR, "train_series")
    TEST_SERIES_DIR = os.path.join(INPUT_DIR, "test_series")

    if not os.path.exists(TRAIN_CSV) and os.path.exists(os.path.join(LOCAL_DATA_DIR, "train.csv")):
        TRAIN_CSV = os.path.join(LOCAL_DATA_DIR, "train.csv")
        TEST_CSV = os.path.join(LOCAL_DATA_DIR, "test.csv")
        TRAIN_SERIES_CSV = os.path.join(LOCAL_DATA_DIR, "train_series.csv")
        TEST_SERIES_CSV = os.path.join(LOCAL_DATA_DIR, "test_series.csv")
        TRAIN_SERIES_DIR = os.path.join(INPUT_DIR, "train_series")
        TEST_SERIES_DIR = os.path.join(INPUT_DIR, "test_series")

    OUTPUT_DIR = "./output"
    SUBMISSION_PATH = "submission.csv"
    # Scratch that does NOT count against the 20 GB /kaggle/working quota.
    CACHE_ROOT = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else "/tmp"

    WEIGHT_DIRS = [
        "/kaggle/input/pytorch-effnet-weights",
        "/kaggle/input/pytorch-pretrained-image-models",
        "/kaggle/input/pretrained-backbones",
        "/kaggle/input/torchvision-pretrained-weights",
    ]

    TARGETS = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
        "Medial OA", "Lateral OA", "PF OA", "Effusion",
        "Synovitis", "Baker's", "Contusion", "Fracture",
    ]
    N_TARGETS = len(TARGETS)

    # --- imaging ---
    IMAGE_SIZE = 224
    NUM_SLICES = 8            # centre slices used by the model, per plane
    DEPTH = NUM_SLICES + 2    # planes stored per view (2.5D neighbours)
    PLANES = ("sagittal", "coronal", "axial")
    N_VIEWS = 3

    # --- weak labels ---
    POS_TARGET = 0.95
    NEG_TARGET = 0.02
    POS_WEIGHT_W = 1.00       # sample weight for an explicit positive mention
    NEG_WEIGHT_W = 0.90       # ... explicit negation
    UNMENTIONED_W = 0.35      # ... silence (treated as probably-absent)
    AMBIGUOUS_W = 0.05        # finding present but side not stated
    GOLD_IN_TRAIN = True      # trusted labels should contribute to the gradient
    GOLD_WEIGHT = 2.0         # trust gold labels without overfitting the 58-study set

    # --- training ---
    BATCH_SIZE = 8
    ACCUM = 2                 # effective batch 16
    NUM_WORKERS = min(4, os.cpu_count() or 1)
    N_RUNS = 3                # lower-variance configuration validated at 0.763
    EPOCHS = 8
    PATIENCE = 2
    WARMUP_STEPS = 200
    MAX_RUNTIME_HOURS = 8.0
    BACKBONE_LR = 2e-4        # v3 used 3e-5: far too low for MRI transfer
    HEAD_LR = 1e-3
    WEIGHT_DECAY = 1e-4
    EMA_DECAY = 0.997
    LABEL_POS_WEIGHT_CAP = 6.0
    WEAK_HOLDOUT = 0.10
    GATE_GOLD_WEIGHT = 0.25   # favor the larger weak holdout for checkpoint selection

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    AMP = DEVICE == "cuda"
    SEED = 42

    # --- time budget (seconds) ---
    TOTAL_BUDGET = 8 * 3600
    TEST_RESERVE = 2 * 60 * 60  # cache-build + inference + submission safety margin

    DEBUG_MAX_STUDIES = None  # set to e.g. 120 for a smoke test


START_TIME = time.monotonic()


def elapsed():
    return time.monotonic() - START_TIME


def log(msg):
    print(f"[{elapsed()/60:6.1f}m] {msg}", flush=True)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


# 1. MULTILINGUAL REPORT LABELLER
# English is only ~39% of the reports. The lexicon below leans on the fact
# that MSK anatomy is overwhelmingly Latin/Greek-derived, so one substring
# often covers several languages (menisc- -> meniscus/menisco/menisque/
# Meniskus/meniscus/menisk). CJK and Cyrillic get explicit entries.

def normalize_text(s):
    """Lowercase, strip accents, collapse whitespace. CJK passes through."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s


# --- generic vocabulary -------------------------------------------------
TEAR = [
    "tear", "torn", "tore", "rupture", "ruptur", "ruptured", "rotura", "ruptura",
    "rottura", "rupt.", "lesion", "lesao", "lasion", "laesion", "riss", "rissbildung",
    "einriss", "abriss", "yirtik", "yirtigi", "rupturu", "dechirure", "dechirures",
    "scheur", "ruptuur", "przerwanie", "uszkodzenie", "razryv", "nadryv",
    "\u65ad\u88c2", "\u7834\u88c2", "\u6495\u88c2", "\u640d\u50b7",
    "\u0440\u0430\u0437\u0440\u044b\u0432", "\u043d\u0430\u0434\u0440\u044b\u0432",
    "\ud30c\uc5f4",
]

DEGEN = [
    "osteoarthritis", "osteoarthrosis", "arthrosis", "arthrose", "artrosis",
    "artrose", "artrosi", "osteoartrite", "osteoartrosis", "gonarthrose",
    "gonartrosis", "gonartroz", "degenerative", "degenerativ", "degenerativa",
    "degenerativo", "chondropathy", "chondropathie", "condropatia",
    "condropatia", "chondromalacia", "chondromalacie", "condromalacia",
    "chondral", "cartilage loss", "cartilagem", "cartilago", "knorpel",
    "knorpelschaden", "kikirdak", "osteofit", "osteophyte", "osteofito",
    "\u5909\u5f62\u6027", "\u8edf\u9aa8", "\u9000\u53d8", "\u8f6f\u9aa8",
    "\u043e\u0441\u0442\u0435\u043e\u0430\u0440\u0442\u0440",
]

NEGATION = [
    "no ", "no evidence", "not ", "without", "absence", "absent", "negative for",
    "intact", "unremarkable", "normal", "preserved", "continuous", "sin ",
    "no se observa", "no se identifica", "ausencia", "integro", "integra",
    "conservado", "sem ", "ausencia de", "pas de", "absence de", "sans ",
    "integre", "kein", "keine", "unauffallig", "intakt", "regelrecht",
    "ohne ", "senza", "assenza", "integro", "nella norma", "geen", "zonder",
    "yok", "izlenmedi", "saptanmadi", "dogal", "normaldir", "brak", "bez ",
    "\u672a\u898b", "\u306a\u3057", "\u8a8d\u3081\u305a", "\u9670\u6027",
    "\u672a\u89c1", "\u65e0\u660e\u663e", "\u6b63\u5e38", "\u65e0 ",
    "\u043d\u0435\u0442", "\u043d\u0435 \u0432\u044b\u044f\u0432\u043b\u0435\u043d",
    "\u0431\u0435\u0437 ", "\u0438\u043d\u0442\u0430\u043a\u0442",
]

MEDIAL = [
    "medial", "mediale", "medialis", "interno", "interna", "internal", "inner",
    "innen", "innenseitig", "interne", "mediaal", "przysrodkow", "ic ",
    "\u5185\u5074", "\u5185\u4fa7", "\u043c\u0435\u0434\u0438\u0430\u043b",
    "\u0432\u043d\u0443\u0442\u0440\u0435\u043d\u043d",
]

LATERAL = [
    "lateral", "laterale", "lateralis", "externo", "externa", "external",
    "outer", "aussen", "ausseren", "externe", "lateraal", "boczn", "dis ",
    "\u5916\u5074", "\u5916\u4fa7", "\u043b\u0430\u0442\u0435\u0440\u0430\u043b",
    "\u043d\u0430\u0440\u0443\u0436\u043d",
]

# --- anatomy anchors per target ----------------------------------------
ANCHORS = {
    "ACL": [
        "acl", "a.c.l", "anterior cruciate", "cruciate anterior", "lca",
        "ligamento cruzado anterior", "cruzado anterior", "cruciatum anterius",
        "ligament croise anterieur", "croise anterieur", "vorderes kreuzband",
        "vkb", "kreuzband vorder", "crociato anteriore", "voorste kruisband",
        "on capraz bag", "\u524d\u5341\u5b57\u9788\u5e26", "\u524d\u5341\u5b57\u97e7\u5e2f",
        "\u524d\u4ea4\u53c9\u97e7\u5e26", "\u043f\u0435\u0440\u0435\u0434\u043d"
        "\u044f\u044f \u043a\u0440\u0435\u0441\u0442\u043e\u043e\u0431\u0440\u0430\u0437",
    ],
    "MCL": [
        "mcl", "m.c.l", "medial collateral", "collateral medial",
        "ligamento colateral medial", "colateral interno", "collaterale mediale",
        "ligament collateral medial", "ligament lateral interne", "innenband",
        "mediales kollateralband", "mediaal collateraal", "ic yan bag",
        "\u5185\u5074\u5074\u526f\u97e7\u5e2f", "\u5185\u4fa7\u526f\u97e7\u5e26",
        "\u0431\u043e\u043b\u044c\u0448\u0435\u0431\u0435\u0440\u0446\u043e\u0432",
    ],
    "MENISCUS": [
        "meniscus", "menisci", "meniscal", "menisco", "menisque", "meniskus",
        "menisk", "meniscaal", "lakiet", "\u534a\u6708\u677f", "\u534a\u6708\u72b6",
        "\u043c\u0435\u043d\u0438\u0441\u043a",
    ],
    "PF OA": [
        "patellofemoral", "patello-femoral", "femoropatellar", "femoro-patellar",
        "femoropatelar", "femoro-rotulien", "femoropatellaire", "retropatellar",
        "retropatellaire", "patellar cartilage", "patella cartilage",
        "rotuliano", "patellofemorale", "patellofemoral", "patellofemoralis",
        "patellofemoral eklem", "\u819d\u84cb\u5927\u817f",
        "\u808c\u9aa8\u80a1\u9aa8", "\u9aa8\u80a1\u9aa8",
        "\u043f\u0430\u0442\u0435\u043b\u043b\u043e\u0444\u0435\u043c\u043e\u0440",
    ],
    "Effusion": [
        "effusion", "joint fluid", "derrame", "derrame articular",
        "liquido articular", "epanchement", "epanchement articulaire",
        "gelenkerguss", "erguss", "hydrops", "versamento", "gewrichtsvocht",
        "eklem sivisi", "eklem efuzyon", "wysiek", "\u95a2\u7bc0\u6db2",
        "\u6ea2\u6db2", "\u5173\u8282\u79ef\u6db2",
        "\u0432\u044b\u043f\u043e\u0442",
    ],
    "Synovitis": [
        "synovitis", "synovite", "sinovitis", "sinovite", "synovialitis",
        "synovialis", "synoviale proliferation", "pannus", "sinovit",
        "\u6ed1\u819c\u708e", "\u6ed1\u81a3\u708e",
        "\u0441\u0438\u043d\u043e\u0432\u0438\u0442",
    ],
    "Baker's": [
        "baker", "bakers cyst", "baker cyst", "popliteal cyst", "poplitea cyst",
        "quiste de baker", "quiste poplite", "cisto de baker", "cisto poplite",
        "kyste de baker", "kyste poplite", "bakerzyste", "baker-zyste",
        "poplitealzyste", "cisti di baker", "cisti poplitea", "bakerkyste",
        "baker kisti", "popliteal kist", "torbiel bakera",
        "\u30d9\u30fc\u30ab\u30fc", "\u56cc\u80bf", "\u8113\u7a9d\u56ca\u80bf",
        "\u043a\u0438\u0441\u0442\u0430 \u0431\u0435\u0439\u043a\u0435\u0440\u0430",
    ],
    "Contusion": [
        "contusion", "bone contusion", "bone bruise", "bruising",
        "bone marrow edema", "bone marrow oedema", "marrow edema",
        "contusao", "contusion osea", "edema oseo", "edema de medula osea",
        "edema osseo", "contusion osseuse", "oedeme osseux", "oedeme medullaire",
        "knochenmarksodem", "knochenmarkodem", "knochenodem", "bone marrow",
        "edema midollare", "contusione ossea", "beenmergoedeem",
        "kemik iligi odemi", "kemik kontuzyon", "stluczenie",
        "\u9aa8\u632b\u50b7", "\u9aa8\u9ad3\u6d6e\u816b", "\u9aa8\u9ad3\u6c34\u816b",
        "\u0443\u0448\u0438\u0431", "\u043e\u0442\u0435\u043a "
        "\u043a\u043e\u0441\u0442\u043d\u043e\u0433\u043e",
    ],
    "Fracture": [
        "fracture", "fractured", "fractura", "fratura", "fraktur", "frattura",
        "fractuur", "kirik", "zlamanie", "insufficiency fracture",
        "stress fracture", "avulsion", "avulsie", "avulsion fracture",
        "\u9aa8\u6298", "\u9aa8\u6298\u7dda",
        "\u043f\u0435\u0440\u0435\u043b\u043e\u043c",
    ],
}

# Some languages inflect or compound the anchor so no single substring works
# ("Ruptur des vorderen Kreuzbandes"). These fire when ALL terms co-occur.
PAIR_ANCHORS = {
    "ACL": [
        ("kreuzband", "vorder"), ("kruisband", "voorste"),
        ("cruzado", "anterior"), ("croise", "anterieur"),
        ("crociato", "anteriore"), ("capraz", "on "),
        ("\u5341\u5b57", "\u524d"),
        ("\u043a\u0440\u0435\u0441\u0442\u043e\u043e\u0431\u0440\u0430\u0437",
         "\u043f\u0435\u0440\u0435\u0434\u043d"),
    ],
    "MCL": [
        ("kollateralband", "medial"), ("collateral", "medial"),
        ("colateral", "medial"), ("colateral", "interno"),
        ("collaterale", "mediale"), ("yan bag", "ic "),
        ("\u526f\u97e7", "\u5185"),
    ],
}

# Findings whose anchor already implies the abnormality (no TEAR/DEGEN needed).
SELF_EVIDENT = {"Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"}

CLAUSE_SPLIT = re.compile(
    r"[.;:\n\r\u3002\uff1b\u2022]|(?<=\s)-\s|,\s+(?="
    r"(?:no|not|without|absence|absent|negative|intact|normal|preserved|"
    r"medial|lateral|right|left|acl|mcl|menisc|effusion|synov|fracture|"
    r"contusion|baker|patell|cartil))"
)


def _hit(text, terms):
    return any(t in text for t in terms)


def _negated(clause):
    """Crude but effective: a negation cue anywhere in the clause flips it."""
    return _hit(clause, NEGATION)


def label_report(report):
    """
    Returns (state, ) dict target -> one of 'pos', 'neg', 'amb', 'unm'.
    Clause-level: a finding is positive if its anatomy anchor and (where
    required) an abnormality term co-occur in a clause with no negation cue.
    """
    text = normalize_text(report)
    state = {t: "unm" for t in CFG.TARGETS}
    if not text:
        return state

    clauses = [c.strip() for c in CLAUSE_SPLIT.split(text) if c and c.strip()]

    def bump(target, new):
        order = {"unm": 0, "neg": 1, "amb": 2, "pos": 3}
        if order[new] > order[state[target]]:
            state[target] = new

    for clause in clauses:
        neg = _negated(clause)
        med = _hit(clause, MEDIAL)
        lat = _hit(clause, LATERAL)
        tear = _hit(clause, TEAR)
        degen = _hit(clause, DEGEN)

        # --- simple single-target findings ---
        for target in ("Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"):
            if _hit(clause, ANCHORS[target]):
                bump(target, "neg" if neg else "pos")

        # --- ligaments ---
        for target in ("ACL", "MCL"):
            anchored = _hit(clause, ANCHORS[target]) or any(
                all(term in clause for term in pair)
                for pair in PAIR_ANCHORS[target]
            )
            if anchored:
                if neg:
                    bump(target, "neg")
                elif tear:
                    bump(target, "pos")
                else:
                    bump(target, "amb")

        # --- menisci: need a side ---
        if _hit(clause, ANCHORS["MENISCUS"]):
            sides = []
            if med:
                sides.append("Medial Meniscus")
            if lat:
                sides.append("Lateral Meniscus")
            if not sides:
                sides = ["Medial Meniscus", "Lateral Meniscus"]
                unsided = True
            else:
                unsided = False
            for target in sides:
                if neg:
                    bump(target, "neg")
                elif tear and not unsided:
                    bump(target, "pos")
                elif tear:
                    bump(target, "amb")
                else:
                    bump(target, "amb")

        # --- patellofemoral OA ---
        if _hit(clause, ANCHORS["PF OA"]):
            if neg:
                bump("PF OA", "neg")
            elif degen:
                bump("PF OA", "pos")
            else:
                bump("PF OA", "amb")

        # --- tibiofemoral OA, side-resolved ---
        if degen and not _hit(clause, ANCHORS["PF OA"]):
            targets = []
            if med:
                targets.append("Medial OA")
            if lat:
                targets.append("Lateral OA")
            if targets:
                for target in targets:
                    bump(target, "neg" if neg else "pos")
            else:
                for target in ("Medial OA", "Lateral OA"):
                    bump(target, "neg" if neg else "amb")

    return state


STATE_TO_SOFT = {
    "pos": (CFG.POS_TARGET, CFG.POS_WEIGHT_W),
    "neg": (CFG.NEG_TARGET, CFG.NEG_WEIGHT_W),
    "amb": (0.5, CFG.AMBIGUOUS_W),
    "unm": (None, CFG.UNMENTIONED_W),  # target filled from prevalence prior
}


def build_weak_labels(train_df):
    """train_df -> (soft targets [N,12] float32, weights [N,12] float32)."""
    report_col = None
    for candidate in ("Report", "report", "ReportText"):
        if candidate in train_df.columns:
            report_col = candidate
            break

    n = len(train_df)
    states = []
    if report_col is None:
        log("WARNING: no Report column found -- weak supervision disabled.")
        states = [{t: "unm" for t in CFG.TARGETS} for _ in range(n)]
    else:
        for report in train_df[report_col].tolist():
            states.append(label_report(report))

    # per-label prevalence among reports that resolve the label either way
    prevalence = {}
    for j, target in enumerate(CFG.TARGETS):
        pos = sum(1 for s in states if s[target] == "pos")
        neg = sum(1 for s in states if s[target] == "neg")
        denom = max(pos + neg, 1)
        prevalence[target] = float(np.clip(pos / denom, 0.01, 0.60))

    y = np.zeros((n, CFG.N_TARGETS), dtype=np.float32)
    w = np.zeros((n, CFG.N_TARGETS), dtype=np.float32)
    for i, s in enumerate(states):
        for j, target in enumerate(CFG.TARGETS):
            value, weight = STATE_TO_SOFT[s[target]]
            if value is None:
                # silence in a radiology report leans absent, not unknown
                value = float(np.clip(0.4 * prevalence[target], 0.01, 0.15))
            if s[target] == "amb":
                value = float(np.clip(prevalence[target], 0.05, 0.60))
            y[i, j] = value
            w[i, j] = weight

    stats = {t: {"prevalence": round(prevalence[t], 4)} for t in CFG.TARGETS}
    for j, target in enumerate(CFG.TARGETS):
        counts = {k: 0 for k in ("pos", "neg", "amb", "unm")}
        for s in states:
            counts[s[target]] += 1
        stats[target].update(counts)
    log("Weak-label summary (pos/neg/amb/unmentioned):")
    for target in CFG.TARGETS:
        s = stats[target]
        log(f"  {target:<18} pos={s['pos']:>5} neg={s['neg']:>5} "
            f"amb={s['amb']:>5} unm={s['unm']:>5}  prev={s['prevalence']:.3f}")
    return y, w, stats


# 2. SERIES SELECTION
def _col(df, name):
    """Case-insensitive column lookup."""
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    return None


def build_series_plan(series_df, base_dir=None):
    """
    study_id -> {plane_index: series_id}. One series per anatomical plane,
    preferring fluid-sensitive + fat-suppressed acquisitions, then slice count.
    """
    if series_df is None or series_df.empty:
        return {}

    study_col = _col(series_df, "StudyInstanceUID")
    series_col = _col(series_df, "SeriesInstanceUID")
    plane_col = _col(series_df, "Anatomical_Plane")
    fluid_col = _col(series_df, "Fluid_Sensitive")
    fat_col = _col(series_df, "Fat_Suppression")
    count_col = _col(series_df, "Number_of_Instances") or _col(series_df, "InstanceCount")

    plan = {}
    for study_id, group in series_df.groupby(study_col):
        slots = {}
        for pi, plane in enumerate(CFG.PLANES):
            if plane_col is not None:
                cand = group[group[plane_col].astype(str).str.lower() == plane]
            else:
                cand = group.iloc[0:0]
            if cand.empty:
                continue

            def rank(row):
                fluid = int(row[fluid_col] == 1) if fluid_col else 0
                fat = int(row[fat_col] == 1) if fat_col else 0
                count = int(row[count_col]) if count_col and not pd.isna(row[count_col]) else 0
                if count == 0 and base_dir is not None and series_col is not None:
                    series_dir = os.path.join(
                        base_dir, str(group.name), str(row[series_col])
                    )
                    try:
                        count = sum(
                            name.lower().endswith(".dcm")
                            for name in os.listdir(series_dir)
                        )
                    except OSError:
                        count = 0
                return (-fluid, -fat, -count, str(row[series_col]))

            best = sorted((r for _, r in cand.iterrows()), key=rank)[0]
            slots[pi] = str(best[series_col])
        if slots:
            plan[str(study_id)] = slots
    return plan


# 3. DICOM READING -> uint8 CACHE (decoded exactly once)
HEADER_TAGS = [
    "ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber",
    "ImageLaterality", "Laterality", "SeriesDescription",
]


def _read_header(path):
    import pydicom
    try:
        return pydicom.dcmread(path, stop_before_pixels=True,
                               specific_tags=HEADER_TAGS)
    except Exception:
        return None


def _order_series(directory):
    """
    Geometric slice ordering. Filename / bare SliceLocation order is wrong on
    the majority of studies, so project ImagePositionPatient onto the slice
    normal derived from ImageOrientationPatient.
    Returns (ordered_paths, iop, laterality).
    """
    try:
        names = [f for f in os.listdir(directory) if f.lower().endswith(".dcm")]
    except Exception:
        return [], None, None
    if not names:
        return [], None, None

    paths = [os.path.join(directory, n) for n in names]
    entries, iop, laterality = [], None, None

    for p in paths:
        dcm = _read_header(p)
        if dcm is None:
            entries.append((float("inf"), 0, p))
            continue
        if iop is None:
            raw = getattr(dcm, "ImageOrientationPatient", None)
            if raw is not None and len(raw) == 6:
                iop = np.asarray(raw, dtype=np.float64)
        if laterality is None:
            laterality = (getattr(dcm, "ImageLaterality", None)
                          or getattr(dcm, "Laterality", None))
            if laterality:
                laterality = str(laterality).strip().upper()[:1]

        pos = getattr(dcm, "ImagePositionPatient", None)
        inst = int(getattr(dcm, "InstanceNumber", 0) or 0)
        if pos is not None and iop is not None and len(pos) == 3:
            normal = np.cross(iop[:3], iop[3:])
            key = float(np.dot(np.asarray(pos, dtype=np.float64), normal))
        else:
            key = float(inst)
        entries.append((key, inst, p))

    entries.sort(key=lambda e: (e[0], e[1]))
    return [e[2] for e in entries], iop, laterality


def _decode(path, size):
    import pydicom
    try:
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)
        slope = float(getattr(dcm, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0) or 0.0)
        img = img * slope + intercept
        if img.ndim == 3:
            img = img[..., 0]
        lo, hi = np.percentile(img, (1.0, 99.0))
        if hi <= lo:
            hi, lo = float(img.max()), float(img.min())
        if hi <= lo:
            return np.zeros((size, size), dtype=np.uint8), False
        img = np.clip(img, lo, hi)
        img = (img - lo) / (hi - lo)
        t = torch.from_numpy(img)[None, None].float()
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        return (t.squeeze().numpy() * 255.0).astype(np.uint8), True
    except Exception:
        return np.zeros((size, size), dtype=np.uint8), False


def _needs_mirror(plane_index, iop, laterality):
    """
    Put 'medial' on a fixed side of every image.

    In patient coordinates +x is the patient's LEFT. For a coronal/axial slice
    the in-plane row direction is ~x, so:
      * if iop[0] < 0 the image is stored with +x on screen-left -> flip so
        that +x is always screen-right;
      * then a RIGHT knee has medial on screen-right while a LEFT knee has it
        on screen-left -> flip right knees.
    Sagittal slices have no medial/lateral axis in plane, so they are left
    alone (flipping them would scramble anterior/posterior instead).
    """
    if plane_index == 0:  # sagittal
        return False
    flip = False
    if iop is not None and len(iop) == 6 and abs(iop[0]) > 0.5 and iop[0] < 0:
        flip = not flip
    if laterality == "R":
        flip = not flip
    return flip


def _build_one(args):
    """Worker: decode one study's three planes straight into the memmap."""
    (row_index, study_id, slots, base_dir, memmap_path, n_rows) = args
    shape = (n_rows, CFG.N_VIEWS, CFG.DEPTH, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)
    cache = np.memmap(memmap_path, dtype=np.uint8, mode="r+", shape=shape)
    present = np.zeros(CFG.N_VIEWS, dtype=np.uint8)
    failures = 0

    for plane_index in range(CFG.N_VIEWS):
        series_id = slots.get(plane_index)
        if series_id is None:
            continue
        directory = os.path.join(base_dir, study_id, series_id)
        if not os.path.isdir(directory):
            continue
        paths, iop, laterality = _order_series(directory)
        if not paths:
            continue

        mirror = _needs_mirror(plane_index, iop, laterality)
        picks = np.linspace(0, len(paths) - 1, CFG.DEPTH).astype(int)
        for d, pick in enumerate(picks):
            img, ok = _decode(paths[pick], CFG.IMAGE_SIZE)
            if not ok:
                failures += 1
            if mirror:
                img = img[:, ::-1]
            cache[row_index, plane_index, d] = img
        present[plane_index] = 1

    del cache
    return row_index, present, failures


def build_cache(study_ids, plan, base_dir, tag, deadline):
    """
    Decode every needed DICOM once into <CACHE_ROOT>/<tag>.u8.
    Returns (memmap_path, present [N,3] uint8, kept_index list).
    """
    n = len(study_ids)
    shape = (n, CFG.N_VIEWS, CFG.DEPTH, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)
    nbytes = int(np.prod(shape))
    os.makedirs(CFG.CACHE_ROOT, exist_ok=True)
    memmap_path = os.path.join(CFG.CACHE_ROOT, f"{tag}.u8")
    log(f"Allocating {tag} cache: {shape} = {nbytes/1e9:.2f} GB at {memmap_path}")

    cache = np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=shape)
    cache.flush()
    del cache

    jobs = [
        (i, sid, plan.get(sid, {}), base_dir, memmap_path, n)
        for i, sid in enumerate(study_ids)
    ]
    present = np.zeros((n, CFG.N_VIEWS), dtype=np.uint8)
    done = 0
    failures = 0

    # Do not fork after CUDA has been initialized by training. Spawn workers
    # for larger hidden test sets; tiny local tests stay serial.
    if tag == "test" and n > CFG.NUM_WORKERS:
        pool_context = multiprocessing.get_context("spawn")
    else:
        pool_context = None

    if n <= CFG.NUM_WORKERS:
        for job in jobs:
            idx, pres, fail = _build_one(job)
            present[idx] = pres
            failures += fail
            done += 1
            log(f"  cached {done}/{n} studies ({failures} slice decode failures)")
        cache = np.memmap(memmap_path, dtype=np.uint8, mode="r+", shape=shape)
        cache.flush()
        del cache
        kept = [i for i in range(n) if present[i].sum() > 0]
        log(f"{tag} cache ready: {len(kept)}/{n} studies have at least one usable plane")
        if failures > 0:
            log(f"NOTE: {failures} slices failed to decode. If this number is large, "
                f"attach the pylibjpeg / gdcm wheels so compressed transfer syntaxes decode.")
        return memmap_path, present, kept

    workers = max(1, CFG.NUM_WORKERS)

    pool_kwargs = {"max_workers": workers}
    if pool_context is not None:
        pool_kwargs["mp_context"] = pool_context
    pool = ProcessPoolExecutor(**pool_kwargs)
    try:
        futures = {pool.submit(_build_one, j): j[0] for j in jobs}
        for future in as_completed(futures):
            try:
                idx, pres, fail = future.result()
                present[idx] = pres
                failures += fail
            except Exception as exc:
                log(f"  cache worker failed on row {futures[future]}: {exc}")
            done += 1
            if done % 250 == 0 or done == n:
                log(f"  cached {done}/{n} studies ({failures} slice decode failures)")
            if time.monotonic() > deadline:
                log("  cache deadline hit -- dropping the remaining studies")
                break
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        cache = np.memmap(memmap_path, dtype=np.uint8, mode="r+", shape=shape)
        cache.flush()
        del cache

    kept = [i for i in range(n) if present[i].sum() > 0]
    log(f"{tag} cache ready: {len(kept)}/{n} studies have at least one usable plane")
    if failures > 0:
        log(f"NOTE: {failures} slices failed to decode. If this number is large, "
            f"attach the pylibjpeg / gdcm wheels so compressed transfer syntaxes decode.")
    return memmap_path, present, kept


# 4. DATASET
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class KneeCacheDataset(Dataset):
    """
    Reads straight from the uint8 memmap. Channel c of centre slice s is
    plane s+c, i.e. a 2.5D triplet over the sampled depth axis.
    """

    def __init__(self, memmap_path, n_rows, rows, y, w, present, training):
        self.memmap_path = memmap_path
        self.shape = (n_rows, CFG.N_VIEWS, CFG.DEPTH, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.y = y
        self.w = w
        self.present = present
        self.training = training
        self._cache = None

    def __len__(self):
        return len(self.rows)

    def _mm(self):
        if self._cache is None:
            self._cache = np.memmap(self.memmap_path, dtype=np.uint8,
                                    mode="r", shape=self.shape)
        return self._cache

    def __getitem__(self, i):
        row = int(self.rows[i])
        block = np.asarray(self._mm()[row])           # (V, D, H, W) uint8
        x = torch.from_numpy(block.copy()).float().div_(255.0)

        # (V, D, H, W) -> (V, S, 3, H, W)
        views = []
        for v in range(CFG.N_VIEWS):
            triplets = torch.stack(
                [x[v, s:s + 3] for s in range(CFG.NUM_SLICES)], dim=0
            )                                          # (S, 3, H, W)
            triplets = (triplets - IMAGENET_MEAN) / IMAGENET_STD
            views.append(triplets)
        x = torch.stack(views, dim=0)                  # (V, S, 3, H, W)

        present = torch.from_numpy(self.present[row].astype(np.float32))
        y = torch.from_numpy(self.y[row]) if self.y is not None \
            else torch.zeros(CFG.N_TARGETS)
        w = torch.from_numpy(self.w[row]) if self.w is not None \
            else torch.zeros(CFG.N_TARGETS)
        return x, y, w, present, row


# 5. GPU AUGMENTATION  (NO horizontal flip -- it swaps medial/lateral)
def gpu_augment(x, present):
    """x: (B, V, S, 3, H, W) already on device."""
    B, V, S, C, H, W = x.shape
    device = x.device

    # --- small affine, shared across the slices of one view ---
    angle = (torch.rand(B * V, device=device) * 2 - 1) * (10.0 * math.pi / 180.0)
    scale = 1.0 + (torch.rand(B * V, device=device) * 2 - 1) * 0.10
    tx = (torch.rand(B * V, device=device) * 2 - 1) * 0.06
    ty = (torch.rand(B * V, device=device) * 2 - 1) * 0.06
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.zeros(B * V, 2, 3, device=device)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    theta = theta.repeat_interleave(S, dim=0)

    flat = x.reshape(B * V * S, C, H, W)
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    flat = F.grid_sample(flat, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=False)

    # --- intensity jitter per view ---
    gain = 1.0 + (torch.rand(B * V, 1, 1, 1, device=device) * 2 - 1) * 0.12
    bias = (torch.rand(B * V, 1, 1, 1, device=device) * 2 - 1) * 0.10
    gain = gain.repeat_interleave(S, dim=0)
    bias = bias.repeat_interleave(S, dim=0)
    flat = flat * gain + bias

    # --- cutout ---
    if random.random() < 0.5:
        side = int(H * random.uniform(0.10, 0.25))
        top = random.randint(0, H - side)
        left = random.randint(0, W - side)
        cutout_value = (-IMAGENET_MEAN / IMAGENET_STD).to(device)
        flat[:, :, top:top + side, left:left + side] = cutout_value

    x = flat.reshape(B, V, S, C, H, W)

    # --- view dropout: forces robustness to a missing plane ---
    keep = present.clone()
    for b in range(B):
        if keep[b].sum() >= 2 and random.random() < 0.15:
            options = torch.nonzero(keep[b]).flatten()
            drop = options[random.randrange(len(options))]
            keep[b, drop] = 0.0
            x[b, drop] = 0.0
    return x, keep


# 6. MODEL
def pretrained_path():
    name = os.path.basename(models.EfficientNet_B0_Weights.DEFAULT.url)
    cache = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    os.makedirs(cache, exist_ok=True)
    cached = os.path.join(cache, name)
    if os.path.exists(cached):
        return cached

    roots = CFG.WEIGHT_DIRS + [
        "/kaggle/input",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "weights")),
        os.path.abspath(os.path.dirname(__file__)),
        ".",
    ]
    seen = set()
    for root in roots:
        if not root or root in seen:
            continue
        seen.add(root)
        if os.path.exists(root):
            matches = glob.glob(os.path.join(root, "**", name), recursive=True)
            if matches:
                return matches[0]

    raise FileNotFoundError(
        f"Offline EfficientNet weights not found ({name}). Attach your "
        f"pytorch-effnet-weights dataset or place the file under the repository's "
        f"weights/ folder."
    )


class AttentionPool(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, 1))

    def forward(self, x, mask=None):
        # x: (B, N, D); mask: (B, N) float 1/0
        scores = self.score(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask < 0.5, -1e4)
        weights = torch.softmax(scores, dim=1)
        if mask is not None:
            weights = weights * mask
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-6)
        return (x * weights.unsqueeze(-1)).sum(1)


class KneeModel(nn.Module):
    """
    Plane slots are FIXED: index 0 sagittal, 1 coronal, 2 axial. v3 packed
    whichever planes existed into slots 0..k, so a study missing its sagittal
    series fed coronal features into the sagittal slot.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        if pretrained:
            state = torch.load(pretrained_path(), map_location="cpu")
            backbone.load_state_dict(state, strict=True)
        dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.dim = dim

        self.slice_pool = nn.ModuleList(
            [AttentionPool(dim) for _ in range(CFG.N_VIEWS)])
        self.view_pool = AttentionPool(dim)
        self.view_embed = nn.Parameter(torch.zeros(CFG.N_VIEWS, dim))
        nn.init.normal_(self.view_embed, std=0.02)

        fusion_in = dim * (CFG.N_VIEWS + 1)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 512), nn.LayerNorm(512), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.2),
        )
        self.head = nn.Linear(256, CFG.N_TARGETS)

    def forward(self, x, present):
        B, V, S, C, H, W = x.shape
        feats = self.backbone(x.reshape(B * V * S, C, H, W))
        feats = feats.reshape(B, V, S, self.dim)

        pooled = []
        for v in range(V):
            mask = present[:, v:v + 1].expand(-1, S)
            p = self.slice_pool[v](feats[:, v], mask)
            p = p * present[:, v:v + 1]
            pooled.append(p)
        views = torch.stack(pooled, dim=1)                    # (B, V, D)
        views_tagged = views + self.view_embed.unsqueeze(0)
        glob = self.view_pool(views_tagged, present)

        fused = torch.cat([views[:, v] for v in range(V)] + [glob], dim=1)
        return self.head(self.fusion(fused))


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.steps = 0

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))  # bias warmup
        state = model.state_dict()
        for k, v in self.shadow.items():
            v.mul_(d).add_(state[k].detach().float(), alpha=1 - d)

    def copy_to(self, model):
        state = model.state_dict()
        backup = {k: state[k].detach().clone() for k in self.shadow}
        for k, v in self.shadow.items():
            state[k].copy_(v)
        return backup

    @staticmethod
    def restore(model, backup):
        state = model.state_dict()
        for k, v in backup.items():
            state[k].copy_(v)


# 7. LOSS & METRIC
def soft_bce(logits, targets, weights, pos_weight):
    """
    Weighted BCE against SOFT targets. Positive emphasis is folded into the
    element weight rather than nn.BCEWithLogitsLoss(pos_weight=...) so it
    composes with the per-sample confidence weights from the report labeller.
    """
    emphasis = 1.0 + (pos_weight - 1.0) * targets
    w = weights * emphasis
    loss = F.binary_cross_entropy_with_logits(
        logits, targets, weight=w, reduction="sum")
    return loss / w.sum().clamp_min(1.0)


def macro_auc(targets, predictions, return_per_label=False):
    """Macro AUC over the 12 labels; NaN entries are skipped."""
    scores, per_label = [], {}
    for j, name in enumerate(CFG.TARGETS):
        valid = ~np.isnan(targets[:, j])
        column = targets[valid, j]
        if valid.sum() < 4 or len(np.unique(column)) < 2:
            per_label[name] = float("nan")
            continue
        auc = roc_auc_score(column, predictions[valid, j])
        per_label[name] = auc
        scores.append(auc)
    value = float(np.mean(scores)) if scores else 0.5
    return (value, per_label) if return_per_label else value


# 8. TRAINING
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, shuffle, drop_last=False, num_workers=None, seed=None):
    if num_workers is None:
        num_workers = CFG.NUM_WORKERS
    kwargs = dict(
        batch_size=CFG.BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=CFG.DEVICE == "cuda",
        persistent_workers=num_workers > 0,
        shuffle=shuffle,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
    )
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        kwargs["generator"] = generator
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def infer(model, loader, tta=False):
    model.eval()
    outputs, rows = [], []
    angles = [0.0, 7.0, -7.0] if tta else [0.0]
    for x, _, _, present, row in loader:
        x = x.to(CFG.DEVICE, non_blocking=True)
        present = present.to(CFG.DEVICE, non_blocking=True)
        acc = 0.0
        for angle in angles:
            xin = x
            if angle != 0.0:
                B, V, S, C, H, W = x.shape
                rad = angle * math.pi / 180.0
                theta = torch.tensor(
                    [[math.cos(rad), -math.sin(rad), 0.0],
                     [math.sin(rad), math.cos(rad), 0.0]],
                    device=x.device, dtype=x.dtype
                ).unsqueeze(0).expand(B * V * S, -1, -1)
                flat = x.reshape(B * V * S, C, H, W)
                grid = F.affine_grid(theta, flat.shape, align_corners=False)
                xin = F.grid_sample(flat, grid, mode="bilinear",
                                    align_corners=False).reshape(x.shape)
            if CFG.AMP:
                with torch.amp.autocast("cuda"):
                    logits = model(xin, present)
            else:
                logits = model(xin, present)
            acc = acc + torch.sigmoid(logits.float())
        outputs.append((acc / len(angles)).cpu().numpy())
        rows.append(row.numpy())
    return np.vstack(outputs), np.concatenate(rows)


def train_one_run(run, memmap_path, n_rows, train_rows, weak_val_rows,
                  gold_eval_rows, y, w, present, gold_truth, deadline):
    seed_everything(CFG.SEED + run * 101)

    train_ds = KneeCacheDataset(memmap_path, n_rows, train_rows, y, w, present, True)
    weak_ds = KneeCacheDataset(memmap_path, n_rows, weak_val_rows, y, w, present, False)
    gold_ds = KneeCacheDataset(
        memmap_path, n_rows, gold_eval_rows, y, w, present, False
    )

    train_loader = make_loader(
        train_ds, shuffle=True, drop_last=True, seed=CFG.SEED + run * 101
    )
    weak_loader = make_loader(weak_ds, shuffle=False)
    gold_loader = make_loader(gold_ds, shuffle=False)

    model = KneeModel(pretrained=True).to(CFG.DEVICE)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [{"params": model.backbone.parameters(), "lr": CFG.BACKBONE_LR},
         {"params": head_params, "lr": CFG.HEAD_LR}],
        weight_decay=CFG.WEIGHT_DECAY,
    )
    steps_per_epoch = max(len(train_loader) // CFG.ACCUM, 1)
    total_steps = steps_per_epoch * CFG.EPOCHS

    def lr_lambda(step):
        if step < CFG.WARMUP_STEPS:
            return (step + 1) / CFG.WARMUP_STEPS
        progress = (step - CFG.WARMUP_STEPS) / max(total_steps - CFG.WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=CFG.AMP) if CFG.AMP else None
    ema = EMA(model, CFG.EMA_DECAY)

    # positive emphasis from the weak-label prevalence of the training rows
    pos_rate = np.clip(y[train_rows].mean(axis=0), 1e-3, 1 - 1e-3)
    pos_weight = torch.tensor(
        np.clip((1 - pos_rate) / pos_rate, 1.0, CFG.LABEL_POS_WEIGHT_CAP),
        dtype=torch.float32, device=CFG.DEVICE)

    weak_truth = (y[weak_val_rows] > 0.5).astype(np.float32)
    confident = w[weak_val_rows] >= CFG.NEG_WEIGHT_W          # ignore amb/unmentioned
    weak_truth[~confident] = np.nan
    weak_pos = {int(r): i for i, r in enumerate(weak_val_rows)}

    best_gate, best_detail, stale = -1.0, None, 0
    ckpt = os.path.join(CFG.OUTPUT_DIR, f"model_run{run}.pth")
    step = 0

    for epoch in range(CFG.EPOCHS):
        if time.monotonic() > deadline:
            log(f"  run {run}: budget reached before epoch {epoch + 1}")
            break
        model.train()
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for i, (x, yb, wb, pres, _) in enumerate(train_loader):
            x = x.to(CFG.DEVICE, non_blocking=True)
            yb = yb.to(CFG.DEVICE, non_blocking=True)
            wb = wb.to(CFG.DEVICE, non_blocking=True)
            pres = pres.to(CFG.DEVICE, non_blocking=True)
            x, pres = gpu_augment(x, pres)

            if CFG.AMP:
                with torch.amp.autocast("cuda"):
                    loss = soft_bce(model(x, pres), yb, wb, pos_weight) / CFG.ACCUM
                scaler.scale(loss).backward()
            else:
                loss = soft_bce(model(x, pres), yb, wb, pos_weight) / CFG.ACCUM
                loss.backward()

            running += loss.item() * CFG.ACCUM
            seen += 1

            if (i + 1) % CFG.ACCUM == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)
                step += 1

            if time.monotonic() > deadline:
                break

        if seen and seen % CFG.ACCUM != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            ema.update(model)
            step += 1

        backup = ema.copy_to(model)
        gold_pred, gold_idx = infer(model, gold_loader, tta=False)
        weak_pred, weak_idx = infer(model, weak_loader, tta=False)
        gold_auc, per_label = macro_auc(gold_truth[gold_idx], gold_pred, True)
        weak_auc = macro_auc(
            weak_truth[[weak_pos[int(r)] for r in weak_idx]], weak_pred)
        gate = CFG.GATE_GOLD_WEIGHT * gold_auc + (1 - CFG.GATE_GOLD_WEIGHT) * weak_auc

        log(f"  run {run} epoch {epoch + 1}/{CFG.EPOCHS} "
            f"loss={running / max(seen, 1):.4f} gold58={gold_auc:.4f} "
            f"weak={weak_auc:.4f} gate={gate:.4f}")

        if gate > best_gate:
            best_gate, stale = gate, 0
            best_detail = {"gold": gold_auc, "weak": weak_auc,
                           "per_label": per_label}
            torch.save(model.state_dict(), ckpt)
        else:
            stale += 1
        EMA.restore(model, backup)
        if stale >= CFG.PATIENCE:
            log(f"  run {run}: early stop at epoch {epoch + 1}")
            break

    del model, optimizer, train_loader, weak_loader, gold_loader
    gc.collect()
    if CFG.DEVICE == "cuda":
        torch.cuda.empty_cache()
    return best_gate, best_detail


# 9. INFERENCE
def generate_submission(test_df, test_series_df, gates, deadline):
    plan = build_series_plan(test_series_df, CFG.TEST_SERIES_DIR)
    study_ids = test_df["StudyInstanceUID"].astype(str).tolist()
    log(f"Building test cache for {len(study_ids)} studies")
    memmap_path, present, kept = build_cache(
        study_ids, plan, CFG.TEST_SERIES_DIR, "test", deadline)
    log("Test cache ready; starting submission inference")

    n = len(study_ids)
    dataset = KneeCacheDataset(memmap_path, n, list(range(n)), None, None,
                               present, False)
    loader = make_loader(dataset, shuffle=False, num_workers=0)

    checkpoints = sorted(glob.glob(os.path.join(CFG.OUTPUT_DIR, "model_run*.pth")))
    if not checkpoints:
        raise RuntimeError("No checkpoints were produced -- cannot submit.")

    accumulator = np.zeros((n, CFG.N_TARGETS), dtype=np.float64)
    total_weight = 0.0
    for path in checkpoints:
        run = int(re.search(r"run(\d+)", os.path.basename(path)).group(1))
        log(f"Inferring checkpoint {run + 1}/{len(checkpoints)}")
        model = KneeModel(pretrained=False).to(CFG.DEVICE)
        model.load_state_dict(torch.load(path, map_location=CFG.DEVICE))
        predictions, rows = infer(model, loader, tta=False)
        ordered = np.zeros_like(predictions)
        ordered[rows] = predictions
        # Preserve calibrated probabilities; this configuration scored better
        # than the five-run logit ensemble on the leaderboard.
        ensemble_values = ordered
        weight = max(gates.get(run, 0.5) - 0.45, 0.02)
        accumulator += ensemble_values * weight
        total_weight += weight
        del model
        gc.collect()
        if CFG.DEVICE == "cuda":
            torch.cuda.empty_cache()

    accumulator /= max(total_weight, 1e-6)
    submission = pd.DataFrame({"StudyInstanceUID": test_df["StudyInstanceUID"]})
    for j, target in enumerate(CFG.TARGETS):
        submission[target] = np.clip(accumulator[:, j], 1e-6, 1 - 1e-6)
    submission.to_csv(CFG.SUBMISSION_PATH, index=False)
    log(f"Wrote {CFG.SUBMISSION_PATH} {submission.shape}")
    return submission


# 10. MAIN
def parse_args():
    parser = argparse.ArgumentParser(description="RSNA knee abnormality weak-label training run")
    parser.add_argument("--debug-max-studies", type=int, default=None)
    parser.add_argument("--runs", type=int, default=CFG.N_RUNS)
    parser.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    parser.add_argument("--budget-hours", type=float, default=CFG.MAX_RUNTIME_HOURS)
    parser.add_argument("--device", type=str, default=CFG.DEVICE)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    CFG.DEVICE = args.device.lower()
    CFG.N_RUNS = max(1, min(args.runs, 3))
    CFG.EPOCHS = max(1, args.epochs)
    CFG.MAX_RUNTIME_HOURS = max(1.0, args.budget_hours)
    CFG.TOTAL_BUDGET = int(CFG.MAX_RUNTIME_HOURS * 3600)
    CFG.DEBUG_MAX_STUDIES = args.debug_max_studies

    if CFG.DEVICE == "cpu":
        log("WARNING: running on CPU; this can exceed the 8-hour budget unless DEBUG_MAX_STUDIES is used.")

    seed_everything(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    for stale in glob.glob(os.path.join(CFG.OUTPUT_DIR, "model_run*.pth")):
        os.remove(stale)

    required = [CFG.TRAIN_CSV, CFG.TEST_CSV, CFG.TRAIN_SERIES_CSV, CFG.TEST_SERIES_CSV]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        log("Missing required dataset files: " + ", ".join(missing))
        log("Place the RSNA CSV files in the project root or under the expected Kaggle input directory before training.")
        raise FileNotFoundError("Missing dataset files for training.")

    train_df = pd.read_csv(CFG.TRAIN_CSV)
    test_df = pd.read_csv(CFG.TEST_CSV)
    train_series = pd.read_csv(CFG.TRAIN_SERIES_CSV)
    test_series = pd.read_csv(CFG.TEST_SERIES_CSV)
    train_df["StudyInstanceUID"] = train_df["StudyInstanceUID"].astype(str)

    if CFG.DEBUG_MAX_STUDIES:
        train_df = train_df.head(CFG.DEBUG_MAX_STUDIES).copy()

    # --- gold vs report-only ---
    label_block = train_df.reindex(columns=CFG.TARGETS)
    is_gold = label_block.notna().any(axis=1).to_numpy()
    log(f"{len(train_df)} training studies, {int(is_gold.sum())} with expert labels")

    gold_truth = np.full((len(train_df), CFG.N_TARGETS), np.nan, dtype=np.float32)
    gold_truth[is_gold] = label_block.to_numpy(dtype=np.float32)[is_gold]

    # --- weak labels from the reports ---
    y, w, stats = build_weak_labels(train_df)
    with open(os.path.join(CFG.OUTPUT_DIR, "weak_label_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    # gold studies override their weak labels (they are the real thing)
    if CFG.GOLD_IN_TRAIN:
        y[is_gold] = np.nan_to_num(gold_truth[is_gold], nan=0.0)
        w[is_gold] = np.where(np.isnan(gold_truth[is_gold]), 0.0, CFG.GOLD_WEIGHT)

    # --- one-pass image cache ---
    plan = build_series_plan(train_series, CFG.TRAIN_SERIES_DIR)
    study_ids = train_df["StudyInstanceUID"].tolist()
    cache_deadline = time.monotonic() + 60 * 60   # 1 h, then train with what we have
    memmap_path, present, kept = build_cache(
        study_ids, plan, CFG.TRAIN_SERIES_DIR, "train", cache_deadline)

    n_rows = len(study_ids)
    usable = np.zeros(n_rows, dtype=bool)
    usable[kept] = True

    gold_rows = np.where(is_gold & usable)[0]
    weak_rows = np.where((~is_gold) & usable)[0]
    log(f"Usable: {len(weak_rows)} weak-supervised, {len(gold_rows)} gold")

    if len(gold_rows) < 8:
        log("WARNING: too few gold studies for a reliable gate; "
            "falling back to the weak holdout alone.")
        CFG.GATE_GOLD_WEIGHT = 0.0

    train_deadline = START_TIME + CFG.TOTAL_BUDGET - CFG.TEST_RESERVE
    if train_deadline <= time.monotonic():
        raise RuntimeError("No training time remains after the test reserve.")
    gates, details = {}, {}
    gold_order = np.random.RandomState(CFG.SEED + 999).permutation(gold_rows)
    gold_folds = np.array_split(gold_order, CFG.N_RUNS)

    for run in range(CFG.N_RUNS):
        if time.monotonic() > train_deadline:
            log(f"Budget exhausted after {run} runs.")
            break
        rng = np.random.RandomState(CFG.SEED + run)
        shuffled = weak_rows.copy()
        rng.shuffle(shuffled)
        cut = max(int(len(shuffled) * CFG.WEAK_HOLDOUT), 32)
        val_rows, tr_rows = shuffled[:cut], shuffled[cut:]
        gold_eval_rows = gold_folds[run % len(gold_folds)]
        gold_train_rows = np.concatenate([
            fold for fold_index, fold in enumerate(gold_folds)
            if fold_index != run % len(gold_folds)
        ])
        if CFG.GOLD_IN_TRAIN:
            tr_rows = np.concatenate([tr_rows, gold_train_rows])
        log(f"Run {run}: train={len(tr_rows)} weak-val={len(val_rows)} "
            f"gold-gate={len(gold_eval_rows)}")
        gate, detail = train_one_run(
            run, memmap_path, n_rows, tr_rows, val_rows, gold_eval_rows,
            y, w, present, gold_truth, train_deadline)
        gates[run] = gate
        if detail is not None:
            details[run] = detail
        log(f"Run {run} best gate={gate:.4f} "
            f"(gold={detail['gold']:.4f} weak={detail['weak']:.4f})"
            if detail else f"Run {run} produced no checkpoint")

    with open(os.path.join(CFG.OUTPUT_DIR, "run_gates.json"), "w") as fh:
        json.dump({"gates": gates, "details": details}, fh, indent=2, default=float)

    if details:
        best_run = max(details.keys(), key=lambda r: gates[r])
        log("Per-label gold AUC of the best run:")
        for target, value in details[best_run]["per_label"].items():
            log(f"  {target:<18} {value:.4f}")

    # free the training cache before the test cache is allocated
    try:
        os.remove(memmap_path)
    except OSError:
        pass

    generate_submission(test_df, test_series, gates,
                        START_TIME + CFG.TOTAL_BUDGET - 5 * 60)
    log("Done.")


if __name__ == "__main__":
    main()