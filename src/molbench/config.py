"""Central configuration: paths, seeds, task registry, model registry."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MOLBENCH_ROOT", Path(__file__).resolve().parents[2]))
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FEATURE_DIR = DATA_PROCESSED / "features"
SPLIT_DIR = DATA_PROCESSED / "splits"
PRETRAINED_PATH = ROOT / "data" / "pretrained" / "mat_pretrained_weights.pt"
RESULTS = ROOT / "results"
RESULTS_TUNING = RESULTS / "tuning"
RESULTS_RAW = RESULTS / "raw"
RESULTS_SUMMARY = RESULTS / "summary"
RESULTS_STATS = RESULTS / "stats"
RESULTS_ATTENTION = RESULTS / "attention"
RESULTS_FIGURES = RESULTS / "figures"
RESULTS_CHECKPOINTS = RESULTS / "checkpoints"
REPORTS = ROOT / "reports"

SEEDS = [42, 123, 456, 789, 1011]
N_FOLDS = 5
VAL_FRACTION = 0.10          # fraction of the non-test portion held out for early stopping
MAX_HEAVY_ATOMS = 100
SPLITS = ["random", "scaffold"]

# Task registry. `file` is relative to DATA_RAW. `type` is "clf" or "reg".
TASKS = {
    "bbbp":           {"file": "bbbp/bbbp.csv",                     "type": "clf", "label": "BBB penetration"},
    "esol":           {"file": "esol/esol.csv",                     "type": "reg", "label": "log solubility (z-scored)"},
    "freesolv":       {"file": "freesolv/freesolv.csv",             "type": "reg", "label": "hydration free energy (z-scored)"},
    "estrogen-alpha": {"file": "estrogen-alpha/estrogen-alpha.csv", "type": "clf", "label": "ER-alpha activity"},
    "estrogen-beta":  {"file": "estrogen-beta/estrogen-beta.csv",   "type": "clf", "label": "ER-beta activity"},
    "metstab-high":   {"file": "mesta-high/mesta-high.csv",         "type": "clf", "label": "high metabolic stability"},
    "metstab-low":    {"file": "mesta-low/mesta-low.csv",           "type": "clf", "label": "low metabolic stability"},
}
TASK_NAMES = list(TASKS)
CLF_TASKS = [t for t, v in TASKS.items() if v["type"] == "clf"]
REG_TASKS = [t for t, v in TASKS.items() if v["type"] == "reg"]

CLF_METRICS = ["roc_auc", "pr_auc", "f1"]
REG_METRICS = ["rmse", "mae", "r2"]
PRIMARY_METRIC = {"clf": "roc_auc", "reg": "rmse"}
HIGHER_IS_BETTER = {"roc_auc": True, "pr_auc": True, "f1": True, "rmse": False, "mae": False, "r2": True}

# Tier-1 model registry (full 5 seeds x 5 folds protocol).
TIER1_MODELS = ["rf", "svm", "gcn", "mat", "mat_nograph", "mat_nodistance", "mat_noattention", "hybrid"]
TIER2_MODELS = ["mat_large_pretrained", "mat_large_scratch"]
TORCH_MODELS = ["gcn", "mat", "mat_nograph", "mat_nodistance", "mat_noattention", "hybrid",
                "mat_large_pretrained", "mat_large_scratch"]

MODEL_LABELS = {
    "rf": "RF (ECFP4)", "svm": "SVM/SVR (Tanimoto)", "gcn": "GCN", "mat": "MAT",
    "mat_nograph": "MAT-NoGraph", "mat_nodistance": "MAT-NoDistance", "mat_noattention": "MAT-NoAttention",
    "hybrid": "ECFP+MAT hybrid", "mat_large_pretrained": "MAT-large (pretrained)",
    "mat_large_scratch": "MAT-large (scratch)",
}

# Featurisation constants (identical to the original MAT code, one-hot formal charge on).
ONE_HOT_FORMAL_CHARGE = True
D_ATOM_BASE = 27                    # 11 + 6 + 5 + 3 + 2
D_ATOM = D_ATOM_BASE + 1            # + dummy-node indicator column
ECFP_RADIUS = 2
ECFP_BITS = 2048
CONFORMER_SEED = 0xC0FFEE           # fixed ETKDG seed for reproducible geometry

# Architecture of the released pretrained checkpoint (verified from the state dict).
PRETRAINED_ARCH = dict(d_model=1024, N=8, h=16, N_dense=1, lambda_attention=0.33, lambda_distance=0.33,
                       leaky_relu_slope=0.1, dense_output_nonlinearity="relu", distance_matrix_kernel="exp",
                       dropout=0.0, aggregation_type="mean")


def ensure_dirs() -> None:
    for d in [DATA_PROCESSED, FEATURE_DIR, SPLIT_DIR, RESULTS_TUNING, RESULTS_RAW, RESULTS_SUMMARY,
              RESULTS_STATS, RESULTS_ATTENTION, RESULTS_FIGURES, RESULTS_CHECKPOINTS, REPORTS]:
        d.mkdir(parents=True, exist_ok=True)
