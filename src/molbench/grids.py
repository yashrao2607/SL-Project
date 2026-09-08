"""Hyperparameter grids (PRD section 3.2). Tuned once per (family, task, split) on seed 42 / fold 0."""
from __future__ import annotations


def rf_grid(task_type: str):
    grid = []
    for mf in ["sqrt", 0.3]:
        for msl in [1, 3]:
            if task_type == "clf":
                for cw in [None, "balanced"]:
                    grid.append(dict(n_estimators=500, max_features=mf, min_samples_leaf=msl, class_weight=cw))
            else:
                grid.append(dict(n_estimators=500, max_features=mf, min_samples_leaf=msl))
    return grid


def svm_grid(task_type: str):
    if task_type == "clf":
        return [dict(C=c, class_weight=cw) for c in [0.1, 1.0, 10.0] for cw in [None, "balanced"]]
    return [dict(C=c, epsilon=e) for c in [0.1, 1.0, 10.0] for e in [0.05, 0.1, 0.2]]


def gcn_grid(task_type: str):
    return [dict(hidden=h, n_layers=l, lr=1e-3, dropout=0.1, batch_size=64) for h in [64, 128] for l in [2, 3]]


def mat_grid(task_type: str):
    # (d_model, N) in {(64, 2), (64, 4), (128, 2)}: the (128, 4) cell costs 3x a (64, 2) run and is dropped
    # to keep the 2800-run CPU protocol tractable (documented in PRD 3.2).
    return [dict(d_model=d, N=n, h=4, lr=lr, dropout=0.1, batch_size=64, distance_matrix_kernel="softmax")
            for (d, n) in [(64, 2), (64, 4), (128, 2)] for lr in [5e-4, 1e-3]]


GRIDS = {"rf": rf_grid, "svm": svm_grid, "gcn": gcn_grid, "mat": mat_grid}

# Families that inherit MAT's tuned configuration instead of being tuned separately.
INHERITS_MAT = {"mat_nograph": "mat", "mat_nodistance": "mat", "mat_noattention": "mat", "hybrid": "mat"}

# Tier-2 (pretrained architecture) fixed configuration.
TIER2_CONFIG = dict(lr=1e-4, batch_size=32, dropout=0.0, max_epochs=15, patience=5)
# Local CPU-only Tier-2 budget (seed 42, fold 0): the full 5-seed protocol runs on a GPU via notebooks/colab_tier2.ipynb.
TIER2_LOCAL = dict(seeds=[42], max_epochs=8, patience=3)
