"""Phase 3.3: Tier-1 benchmark - 8 models x 7 tasks x 2 splits x 5 seeds x 5 folds = 2800 runs.

Usage: python scripts/03_run_tier1.py [--models ...] [--tasks ...] [--workers 8] [--threads 2]
"""
from __future__ import annotations

import argparse
import json

from molbench import config as C
from molbench.grids import INHERITS_MAT
from molbench.runner import run_grid


def resolve_config(best: dict, model: str, task: str, split: str) -> dict:
    fam = INHERITS_MAT.get(model, model)
    return best[fam][task][split]


def build_jobs(models, tasks, best, max_epochs, patience, seeds=C.SEEDS, folds=range(C.N_FOLDS)):
    jobs = []
    for model in models:
        for task in tasks:
            for split in C.SPLITS:
                cfg = resolve_config(best, model, task, split)
                for seed in seeds:
                    for fold in folds:
                        jobs.append({"model": model, "task": task, "split": split, "seed": seed, "fold": fold,
                                     "config": cfg, "max_epochs": max_epochs, "patience": patience})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=C.TIER1_MODELS)
    ap.add_argument("--tasks", nargs="+", default=C.TASK_NAMES)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()
    C.ensure_dirs()
    best = json.loads((C.RESULTS_TUNING / "best_configs.json").read_text())
    jobs = build_jobs(args.models, args.tasks, best, args.max_epochs, args.patience)
    jobs.sort(key=lambda j: (j["model"] not in ("rf", "svm"), j["task"], j["split"], j["seed"], j["fold"]))
    run_grid(jobs, C.RESULTS_RAW / "tier1_runs.csv", workers=args.workers, threads=args.threads)


if __name__ == "__main__":
    main()
