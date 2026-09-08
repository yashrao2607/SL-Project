"""Parallel, resume-safe execution of run grids (multiprocessing across CPU cores)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from multiprocessing import Pool

import pandas as pd

from . import config as C
from . import data as D
from . import featurize as Fz
from .engine import run_one

ROW_FIELDS = ["run_key", "tag", "model", "task", "task_type", "split", "seed", "fold", "roc_auc", "pr_auc", "f1", "rmse",
              "mae", "r2", "val_roc_auc", "val_pr_auc", "val_f1", "val_rmse", "val_mae", "val_r2", "best_epoch",
              "epochs_run", "n_params", "wall_time_s", "n_train", "n_val", "n_test", "config", "cfg_hash",
              "max_epochs", "patience", "device", "torch_version", "error"]

_CACHE = None
_TASK_DATA = {}
_THREADS = 2


def cfg_hash(config: dict, max_epochs: int, patience: int) -> str:
    """Short hash of the full training configuration so resumed grids never reuse stale rows."""
    payload = json.dumps({"config": config, "max_epochs": max_epochs, "patience": patience}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def job_hash(job: dict) -> str:
    return cfg_hash(job["config"], job.get("max_epochs", 100), job.get("patience", 15))


def run_key(model, task, split, seed, fold, tag="", chash=""):
    return f"{model}|{task}|{split}|{seed}|{fold}|{tag}|{chash}"


def _init_worker(threads: int):
    global _CACHE, _THREADS
    _THREADS = threads
    try:  # limit BLAS/OpenMP pools of this worker (numpy / scikit-learn kernels); torch is limited per run
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=threads)
    except Exception:
        pass
    import torch
    torch.set_num_threads(threads)
    _CACHE = Fz.load_cache()


def _task_data(task):
    if task not in _TASK_DATA:
        _TASK_DATA[task] = Fz.task_features(task, _CACHE)
    return _TASK_DATA[task]


def run_job(job: dict) -> dict:
    """Execute one job dict {model, task, split, seed, fold, config, max_epochs, patience}."""
    global _CACHE
    if _CACHE is None:                     # sequential (non-pool) use
        _init_worker(job.get("threads", _THREADS))
    recs, X_fp, y, ttype = _task_data(job["task"])
    folds = D.load_splits(job["task"], job["split"], job["seed"])
    fold = folds[job["fold"]]
    from .engine import DEVICE
    import torch
    chash = job_hash(job)
    row = {"run_key": run_key(job["model"], job["task"], job["split"], job["seed"], job["fold"], job.get("tag", ""), chash),
           "tag": job.get("tag", ""), "model": job["model"], "task": job["task"], "task_type": ttype, "split": job["split"],
           "seed": job["seed"], "fold": job["fold"], "n_train": len(fold["train"]), "n_val": len(fold["val"]),
           "n_test": len(fold["test"]), "config": json.dumps(job["config"], sort_keys=True), "cfg_hash": chash,
           "max_epochs": job.get("max_epochs", 100), "patience": job.get("patience", 15), "device": str(DEVICE),
           "torch_version": torch.__version__, "error": ""}
    try:
        res = run_one(job["model"], job["config"], recs, X_fp, y, ttype, fold, job["seed"],
                      max_epochs=job.get("max_epochs", 100), patience=job.get("patience", 15), n_threads=_THREADS)
        row.update({k: v for k, v in res.items() if k in ROW_FIELDS})
    except Exception as e:  # keep the grid going; the row records the failure explicitly
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def load_done(csv_path) -> set:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return set()
    df = pd.read_csv(csv_path)
    if "error" in df:
        df = df[df["error"].isna() | (df["error"].astype(str) == "")]
    return set(df["run_key"])


def run_grid(jobs: list[dict], csv_path, workers: int = 8, threads: int = 2, verbose: bool = True) -> None:
    done = load_done(csv_path)
    todo = [j for j in jobs if run_key(j["model"], j["task"], j["split"], j["seed"], j["fold"], j.get("tag", ""), job_hash(j)) not in done]
    if verbose:
        print(f"[grid] {len(jobs)} jobs, {len(done)} done, {len(todo)} to run, workers={workers} threads={threads}")
    if not todo:
        return
    new_file = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    t0 = time.time()
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
        if new_file:
            w.writeheader()
            fh.flush()
        if workers <= 1:
            _init_worker(threads)
            it = map(run_job, todo)
        else:
            pool = Pool(workers, initializer=_init_worker, initargs=(threads,))
            it = pool.imap_unordered(run_job, todo)
        for i, row in enumerate(it):
            w.writerow({k: row.get(k, "") for k in ROW_FIELDS})
            fh.flush()
            if verbose and ((i + 1) % 10 == 0 or i == len(todo) - 1):
                el = time.time() - t0
                print(f"[grid] {i + 1}/{len(todo)} done, {el / 60:.1f} min elapsed, eta {el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min")
        if workers > 1:
            pool.close()
            pool.join()
