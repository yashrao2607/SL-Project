"""Phase 3.4: Tier-2 - pretrained MAT vs scratch MAT at the released architecture (42M parameters).

Reduced protocol (PRD section 3.3): 5 seeds x fold 0 x 2 splits x 7 tasks for each of the two variants.

Usage: python scripts/04_run_tier2.py [--seeds 42 123 456 789 1011] [--workers 2] [--threads 8]
"""
from __future__ import annotations

import argparse

from molbench import config as C
from molbench.grids import TIER2_CONFIG, TIER2_LOCAL
from molbench.runner import run_grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=C.SEEDS)
    ap.add_argument("--local", action="store_true", help="CPU budget: seed 42 only, 8 epochs, patience 3 (PRD 3.3)")
    ap.add_argument("--folds", nargs="+", type=int, default=[0])
    ap.add_argument("--tasks", nargs="+", default=C.TASK_NAMES)
    ap.add_argument("--models", nargs="+", default=C.TIER2_MODELS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-epochs", type=int, default=TIER2_CONFIG["max_epochs"])
    ap.add_argument("--patience", type=int, default=TIER2_CONFIG["patience"])
    args = ap.parse_args()
    if args.local:
        args.seeds, args.max_epochs, args.patience = TIER2_LOCAL["seeds"], TIER2_LOCAL["max_epochs"], TIER2_LOCAL["patience"]
    C.ensure_dirs()
    cfg = {k: v for k, v in TIER2_CONFIG.items() if k not in ("max_epochs", "patience")}
    jobs = []
    for task in args.tasks:
        for split in C.SPLITS:
            for seed in args.seeds:
                for fold in args.folds:
                    for model in args.models:
                        jobs.append({"model": model, "task": task, "split": split, "seed": seed, "fold": fold,
                                     "config": cfg, "max_epochs": args.max_epochs, "patience": args.patience})
    run_grid(jobs, C.RESULTS_RAW / "tier2_runs.csv", workers=args.workers, threads=args.threads)


if __name__ == "__main__":
    main()
