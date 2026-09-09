"""Fetch the outputs of the Kaggle kernels and merge them into results/raw/*.csv (de-duplicated by run_key).

    python scripts/fetch_kaggle_results.py [--kernels dipurao/sl-project dipurao/sl-project-t1a ...] [--no-fetch]

Rows are merged by `run_key` (model | task | split | seed | fold | tag | configuration hash); when the same key
exists locally and on Kaggle the local row is kept, so merging is idempotent and never duplicates a run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

from molbench import config as C
from molbench.grids import INHERITS_MAT
from molbench.runner import ROW_FIELDS, cfg_hash

DEFAULT_KERNELS = ["dipurao/sl-project", "dipurao/sl-project-t1a", "dipurao/sl-project-t1d",
                   "dipuraooooo/slproj-b", "dipuraooooo/slproj-c"]
# Kaggle credential directory per account (None = default ~/.kaggle)
USER_CONFIG = {"dipurao": None, "dipuraooooo": "C:/Users/yashr/.kaggle2"}


def kaggle_env(ref: str) -> dict:
    import os
    env = dict(os.environ)
    cfg = USER_CONFIG.get(ref.split("/")[0])
    if cfg:
        env["KAGGLE_CONFIG_DIR"] = cfg
    return env


def merge_into(target: Path, new: pd.DataFrame) -> tuple[int, int]:
    new = new[new["error"].isna() | (new["error"].astype(str) == "")]
    if target.exists() and target.stat().st_size > 0:
        old = pd.read_csv(target)
        old_ok = old[old["error"].isna() | (old["error"].astype(str) == "")]
        add = new[~new["run_key"].isin(set(old_ok["run_key"]))]
        merged = pd.concat([old_ok, add], ignore_index=True)
    else:
        add, merged = new, new
    merged = merged.drop_duplicates("run_key", keep="first")
    merged = merged.reindex(columns=ROW_FIELDS)
    merged.to_csv(target, index=False)
    return len(add), len(merged)


def canonicalise_configs(kernel_best: dict, tasks_in_kernel: set) -> None:
    """Kaggle's tuning choice is canonical for the tasks a kernel processed; merge into results/tuning/best_configs.json."""
    path = C.RESULTS_TUNING / "best_configs.json"
    best = json.loads(path.read_text()) if path.exists() else {}
    for fam, per_task in kernel_best.items():
        for task, per_split in per_task.items():
            if task in tasks_in_kernel:
                best.setdefault(fam, {})[task] = per_split
    path.write_text(json.dumps(best, indent=2))


def filter_to_canonical(target: Path, max_epochs: int = 60, patience: int = 10) -> int:
    """Drop Tier-1 rows whose configuration hash differs from the canonical frozen configuration."""
    path = C.RESULTS_TUNING / "best_configs.json"
    if not target.exists() or not path.exists():
        return 0
    best = json.loads(path.read_text())
    df = pd.read_csv(target)
    keep = []
    for _, r in df.iterrows():
        fam = INHERITS_MAT.get(r["model"], r["model"])
        cfg = best.get(fam, {}).get(r["task"], {}).get(r["split"])
        keep.append(cfg is not None and cfg_hash(cfg, int(r["max_epochs"]), int(r["patience"])) == r["cfg_hash"])
    dropped = int((~pd.Series(keep)).sum())
    if dropped:
        df[pd.Series(keep).values].to_csv(target, index=False)
    return dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", nargs="+", default=DEFAULT_KERNELS)
    ap.add_argument("--no-fetch", action="store_true", help="only merge what is already in kaggle/output/<slug>/")
    args = ap.parse_args()
    C.ensure_dirs()
    for ref in args.kernels:
        slug = ref.split("/")[-1]
        out = C.ROOT / "kaggle" / "output" / slug
        if not args.no_fetch:
            out.mkdir(parents=True, exist_ok=True)
            st = subprocess.run(["kaggle", "kernels", "status", ref], capture_output=True, text=True, env=kaggle_env(ref),
                                encoding="utf-8", errors="replace")
            print(f"[{slug}] {st.stdout.strip()}")
            r = subprocess.run(["kaggle", "kernels", "output", ref, "-p", str(out)], capture_output=True, text=True,
                               env=kaggle_env(ref), encoding="utf-8", errors="replace")
            if r.returncode != 0:
                print(f"[{slug}] fetch failed: {r.stderr.strip()[-300:]}")
        bc = out / "best_configs.json"
        if bc.exists():
            kernel_best = json.loads(bc.read_text())
            tasks = set()
            for f in ["tier1_runs.csv", "tier1_random.csv", "tier1_scaffold.csv"]:
                if (out / f).exists() and (out / f).stat().st_size > 0:
                    tasks |= set(pd.read_csv(out / f)["task"].unique())
            if tasks:
                canonicalise_configs(kernel_best, tasks)
                print(f"[{slug}] canonical configurations adopted for {sorted(tasks)}")
        for name in ["tier1_runs.csv", "tier2_runs.csv", "tier1_random.csv", "tier1_scaffold.csv", "tier2_random.csv", "tier2_scaffold.csv"]:
            f = out / name
            if not f.exists() or f.stat().st_size == 0:
                continue
            df = pd.read_csv(f)
            tier = "tier1" if name.startswith("tier1") else "tier2"
            added, total = merge_into(C.RESULTS_RAW / f"{tier}_runs.csv", df)
            print(f"[{slug}] {name}: {len(df)} rows read, {added} new rows merged -> {tier}_runs.csv now {total} rows")
        prog = out / "progress.txt"
        if prog.exists():
            print(f"[{slug}] progress: " + prog.read_text().strip().splitlines()[-1])
    # Results produced elsewhere (e.g. Colab) and dropped into kaggle/output/colab_*/ are merged too.
    for out in sorted((C.ROOT / "kaggle" / "output").glob("colab_*")):
        for name in ["tier1_runs.csv", "tier2_runs.csv"]:
            f = out / name
            if f.exists() and f.stat().st_size > 0:
                df = pd.read_csv(f)
                tier = "tier1" if name.startswith("tier1") else "tier2"
                added, total = merge_into(C.RESULTS_RAW / f"{tier}_runs.csv", df)
                print(f"[{out.name}] {name}: {len(df)} rows read, {added} new rows merged -> {tier}_runs.csv now {total} rows")
    dropped = filter_to_canonical(C.RESULTS_RAW / "tier1_runs.csv")
    if dropped:
        print(f"[merge] dropped {dropped} tier1 rows trained under non-canonical configurations")
    if (C.RESULTS_RAW / "tier1_runs.csv").exists():
        t1 = pd.read_csv(C.RESULTS_RAW / "tier1_runs.csv")
        print("[merge] tier1 rows per task:", t1.groupby("task").size().to_dict(), "total", len(t1))


if __name__ == "__main__":
    main()
