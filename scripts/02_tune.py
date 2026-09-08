"""Phase 3.2: hyperparameter selection on seed 42 / fold 0 validation sets (PRD section 3.2).

Every grid configuration of RF, SVM/SVR, GCN and MAT is trained on the training portion of
seed-42 fold-0 for every (task, split) and scored on the corresponding validation set. The best
configuration per (family, task, split) is frozen in results/tuning/best_configs.json.

Usage: python scripts/02_tune.py [--workers 8] [--threads 2] [--families rf svm gcn mat]
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from molbench import config as C
from molbench.grids import GRIDS
from molbench.runner import run_grid

TUNE_SEED = 42
TUNE_FOLD = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--families", nargs="+", default=list(GRIDS))
    ap.add_argument("--tasks", nargs="+", default=C.TASK_NAMES)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()
    C.ensure_dirs()
    csv_path = C.RESULTS_TUNING / "tuning_runs.csv"
    jobs = []
    for fam in args.families:
        for task in args.tasks:
            ttype = C.TASKS[task]["type"]
            for split in C.SPLITS:
                for i, cfg in enumerate(GRIDS[fam](ttype)):
                    jobs.append({"model": fam, "task": task, "split": split, "seed": TUNE_SEED, "fold": TUNE_FOLD,
                                 "config": cfg, "tag": f"cfg{i}", "max_epochs": args.max_epochs,
                                 "patience": args.patience})
    # Cheap classical jobs first so the pool stays busy with deep jobs at the end.
    jobs.sort(key=lambda j: (j["model"] not in ("rf", "svm"), j["model"]))
    run_grid(jobs, csv_path, workers=args.workers, threads=args.threads)

    df = pd.read_csv(csv_path)
    df = df[df["error"].isna() | (df["error"].astype(str) == "")]
    best_path = C.RESULTS_TUNING / "best_configs.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else {}
    rows = []
    for fam in args.families:
        best.setdefault(fam, {})
        for task in args.tasks:
            ttype = C.TASKS[task]["type"]
            metric = "val_" + C.PRIMARY_METRIC[ttype]
            best[fam].setdefault(task, {})
            for split in C.SPLITS:
                sub = df[(df["model"] == fam) & (df["task"] == task) & (df["split"] == split)]
                if sub.empty:
                    continue
                idx = sub[metric].idxmax() if C.HIGHER_IS_BETTER[C.PRIMARY_METRIC[ttype]] else sub[metric].idxmin()
                cfg = json.loads(sub.loc[idx, "config"])
                best[fam][task][split] = cfg
                rows.append({"family": fam, "task": task, "split": split, "n_configs": len(sub),
                             metric: float(sub.loc[idx, metric]), "config": json.dumps(cfg, sort_keys=True)})
    best_path.write_text(json.dumps(best, indent=2))
    pd.DataFrame(rows).to_csv(C.RESULTS_TUNING / "best_configs.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main()
