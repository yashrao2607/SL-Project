"""Phase 1: clean the seven tasks, generate splits, featurise every molecule (ECFP + ETKDG geometry).

Usage: python scripts/01_prepare_data.py [--jobs 12]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

from molbench import config as C
from molbench import data as D
from molbench import featurize as Fz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    C.ensure_dirs()
    t0 = time.time()
    logs = {}
    all_smiles = set()
    for task in C.TASK_NAMES:
        ttype = C.TASKS[task]["type"]
        raw = D.load_raw(task)
        clean, log = D.clean_dataset(raw, ttype)
        clean.to_csv(D.processed_path(task), index=False)
        scale = D.recover_regression_scale(task, clean)
        if scale is not None:
            log["original_units"] = scale
        n_folds_info = {}
        for split in C.SPLITS:
            for seed in C.SEEDS:
                folds = D.make_splits(clean, ttype, split, seed)
                D.check_splits(clean, folds, split)
                D.save_splits(task, split, seed, folds)
                n_folds_info[f"{split}_seed{seed}_test_sizes"] = [len(f["test"]) for f in folds]
        log["fold_sizes"] = n_folds_info
        logs[task] = log
        all_smiles |= set(clean["smiles"])
        print(f"[clean] {task:15s} raw={log['n_raw']:4d} clean={log['n_clean']:4d} scaffolds={log['n_scaffolds']:4d} "
              f"dups_removed={log['n_duplicate_rows_removed']} conflicts={log['n_conflicting_label_molecules_dropped']} "
              f"too_large={log['n_too_large']}")
    print(f"[featurize] unique molecules across tasks: {len(all_smiles)}")
    cache = Fz.featurize_all(sorted(all_smiles), n_jobs=args.jobs)
    conf = Counter(cache[s]["conformer"] for s in all_smiles)
    logs["_featurization"] = {"n_unique_molecules": len(all_smiles), "conformer_methods": dict(conf),
                              "d_atom": int(cache[next(iter(all_smiles))]["afm"].shape[1]),
                              "ecfp_bits": C.ECFP_BITS, "seconds": round(time.time() - t0, 1)}
    (C.DATA_PROCESSED / "cleaning_log.json").write_text(json.dumps(logs, indent=2))
    print(f"[done] conformers: {dict(conf)}  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
