"""Phase 4.2: attention analysis of trained MAT models (seed 42, fold 0) on BBBP and ESOL.

Trains MAT with the frozen tuned configuration, captures per-head attention on the test molecules
and writes:
  results/attention/<task>_<split>_head_stats.csv
  results/attention/<task>_<split>_received_by_class.csv
  results/attention/<task>_<split>_uniform_baseline.json
  results/attention/<task>_<split>_examples.npz     (attention maps of 6 example molecules for plotting)
  results/checkpoints/mat_<task>_<split>_seed42_fold0.pt
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from molbench import attention as A
from molbench import config as C
from molbench import data as D
from molbench import featurize as Fz
from molbench.engine import run_torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["bbbp", "esol"])
    ap.add_argument("--splits", nargs="+", default=["scaffold", "random"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    C.ensure_dirs()
    best = json.loads((C.RESULTS_TUNING / "best_configs.json").read_text())
    cache = Fz.load_cache()
    for task in args.tasks:
        recs, X_fp, y, ttype = Fz.task_features(task, cache)
        df = D.load_processed(task)
        for split in args.splits:
            cfg = best["mat"][task][split]
            fold = D.load_splits(task, split, args.seed)[args.fold]
            print(f"[attention] training MAT on {task}/{split} cfg={cfg}")
            res = run_torch("mat", cfg, recs, X_fp, y, ttype, fold, args.seed, n_threads=args.threads,
                            return_model=True)
            model = res["model"]
            ckpt = C.RESULTS_CHECKPOINTS / f"mat_{task}_{split}_seed{args.seed}_fold{args.fold}.pt"
            torch.save({"state_dict": model.state_dict(), "config": cfg, "task": task, "split": split,
                        "metrics": {k: v for k, v in res.items() if k not in ("model", "history", "test_pred", "test_true")}},
                       ckpt)
            te = fold["test"]
            items = A.collect_attention(model, [recs[i] for i in te], X_fp[te], df["smiles"].values[te].tolist())
            stats = A.head_statistics(items)
            stats.insert(0, "task", task)
            stats.insert(1, "split", split)
            stats.to_csv(C.RESULTS_ATTENTION / f"{task}_{split}_head_stats.csv", index=False)
            rec = A.received_by_class(items)
            rec.insert(0, "task", task)
            rec.insert(1, "split", split)
            rec.to_csv(C.RESULTS_ATTENTION / f"{task}_{split}_received_by_class.csv", index=False)
            base = A.baseline_uniform_shares(items)
            base.update({"test_metrics": {k: v for k, v in res.items() if k in C.CLF_METRICS + C.REG_METRICS},
                         "n_test_molecules": len(items), "config": cfg})
            (C.RESULTS_ATTENTION / f"{task}_{split}_uniform_baseline.json").write_text(json.dumps(base, indent=2))
            # example molecules: 6 medium-sized test molecules with the largest |prediction| confidence
            sizes = np.array([it["n"] for it in items])
            cand = np.where((sizes >= 10) & (sizes <= 22))[0][:6]
            np.savez_compressed(C.RESULTS_ATTENTION / f"{task}_{split}_examples.npz",
                                smiles=np.array([items[i]["smiles"] for i in cand]),
                                self_attn=np.array([np.stack(items[i]["self_attn"]) for i in cand], dtype=object),
                                attn=np.array([np.stack(items[i]["attn"]) for i in cand], dtype=object),
                                adj=np.array([items[i]["adj"] for i in cand], dtype=object),
                                dist=np.array([items[i]["dist"] for i in cand], dtype=object),
                                pred=np.array([res["test_pred"][i] for i in cand]),
                                true=np.array([res["test_true"][i] for i in cand]))
            print(stats.groupby("kind")[["bonded_share", "self_share", "dummy_share", "dist_corr_spearman", "entropy"]].mean())
            print("uniform baseline:", {k: round(v, 3) for k, v in base.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
