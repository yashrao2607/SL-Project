"""Kaggle queue scheduler: keeps 2 GPU sessions busy, uploads the finished local tuning first, fetches outputs.

    python kaggle/scheduler.py            (run detached; log in kaggle/scheduler_log.txt)
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--user", default="dipurao")
ap.add_argument("--config-dir", default=None, help="KAGGLE_CONFIG_DIR for this account")
ap.add_argument("--all", nargs="+", default=["sl-project", "sl-project-t1a", "sl-project-t1b", "sl-project-t1c", "sl-project-t1d"])
ap.add_argument("--queue", nargs="*", default=["sl-project-t1b", "sl-project-t1c", "sl-project-t1d"])
ap.add_argument("--stage", default="bash kaggle/stage_dataset.sh 'full local tuning table'",
                help="shell command run once local tuning is complete (empty string = skip)")
ap.add_argument("--stage-script", default=None, help="bash script path used instead of --stage (avoids quoting issues)")
ap.add_argument("--log", default="kaggle/scheduler_log.txt")
ap.add_argument("--assume-staged", action="store_true", help="dataset already uploaded; only wait until its files are listed")
ARGS = ap.parse_args()
if ARGS.stage_script:
    ARGS.stage = f"bash {ARGS.stage_script} 'full local tuning table'"
if ARGS.config_dir:
    os.environ["KAGGLE_CONFIG_DIR"] = ARGS.config_dir
os.environ["PYTHONUTF8"] = "1"                 # the Kaggle CLI prints non-cp1252 glyphs; never let that crash an upload
os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / ARGS.log
USER = ARGS.user
ALL = ARGS.all
QUEUE = ARGS.queue
MAX_SESSIONS = 2
POLL = 120


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


GIT_BASH = r"C:\Program Files\Git\usr\bin\bash.exe"


def sh(cmd, timeout=1800):
    if isinstance(cmd, str):          # shell command -> Git Bash (never WSL bash), inherits the Windows PATH
        cmd = [GIT_BASH, "-lc", cmd]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return 124, f"timeout after {timeout}s: {(e.stdout or '')[-300:] if isinstance(e.stdout, str) else ''}"
    return r.returncode, ((r.stdout or "") + (r.stderr or "")).strip()


def dataset_has(ref, needle, tries=60):
    """Poll `kaggle datasets files` until the newest version lists a file containing `needle` (max ~20 min)."""
    for _ in range(tries):
        code, out = sh(["kaggle", "datasets", "files", ref, "--page-size", "200"])
        if code == 0 and needle in out:
            return True
        time.sleep(20)
    return False


def status(slug):
    code, out = sh(["kaggle", "kernels", "status", f"{USER}/{slug}"])
    if "404" in out or code != 0:
        return "NOT_PUSHED"
    for s in ["RUNNING", "QUEUED", "COMPLETE", "ERROR", "CANCEL"]:
        if s in out.upper():
            return s
    return out[-40:]


def local_tuning_done():
    # run_local_share.bat starts Tier 1 (and its log) only after 02_tune.py has finished and written best_configs.json
    return (ROOT / "results" / "tuning" / "best_configs.json").exists() and (ROOT / "results" / "local_tier1_log.txt").exists()


def main():
    staged = False
    if ARGS.assume_staged:
        staged = dataset_has(f"{USER}/molbench-bundle", "best_configs.json")
        log(f"assume-staged: dataset version with tuning table visible: {staged}")
    fetched = {}                                   # slug -> status at the time of the fetch
    last_push = {}                                 # slug -> time of our push (status lags a few minutes behind)
    GRACE = 420
    pushed = {s for s in ALL if status(s) in ("RUNNING", "QUEUED", "COMPLETE")}   # ERROR kernels are re-pushed
    log(f"start; already running/complete: {sorted(pushed)}")
    while True:
        st = {s: status(s) for s in ALL}
        for s, t in last_push.items():             # a freshly pushed kernel is running even if Kaggle still reports the old state
            if time.time() - t < GRACE and st[s] not in ("RUNNING", "QUEUED"):
                st[s] = "QUEUED"
        active = [s for s in ALL if st[s] in ("RUNNING", "QUEUED")]
        log("status " + json.dumps(st) + f" active={len(active)}")
        if not staged and local_tuning_done():
            if ARGS.stage:
                log("local tuning complete -> " + ARGS.stage)
                code, out = sh(ARGS.stage, timeout=3600)
                log(f"stage exit {code}: {out[-300:]}")
                if code == 0:
                    ok = dataset_has(f"{USER}/molbench-bundle", "best_configs.json")
                    log(f"dataset version visible with tuning table: {ok}")
                    staged = ok
            else:
                staged = True
        queue = [s for s in QUEUE if s not in pushed]
        if staged and queue and len(active) < MAX_SESSIONS:
            nxt = queue[0]
            code, out = sh(["kaggle", "kernels", "push", "-p", f"kaggle/{nxt}"])
            log(f"push {nxt}: exit {code}: {out[-200:]}")
            if code == 0 and "error" not in out.lower():
                pushed.add(nxt)
                last_push[nxt] = time.time()
        for s in ALL:
            if st[s] in ("COMPLETE", "ERROR") and fetched.get(s) != st[s] and s in pushed:
                out_dir = ROOT / "kaggle" / "output" / s
                out_dir.mkdir(parents=True, exist_ok=True)
                code, out = sh(["kaggle", "kernels", "output", f"{USER}/{s}", "-p", str(out_dir)], timeout=3600)
                log(f"fetched {s} ({st[s]}): exit {code}; files: {[p.name for p in out_dir.iterdir()][:12]}")
                fetched[s] = st[s]
                if st[s] == "ERROR":
                    pushed.discard(s)          # allow a re-push if it is in the queue
                prog = out_dir / "progress.txt"
                if prog.exists():
                    log(f"{s} progress tail: " + " | ".join(prog.read_text().strip().splitlines()[-3:]))
                if st[s] == "ERROR":
                    err = out_dir / "error.txt"
                    log(f"{s} ERROR details: " + (err.read_text()[-800:] if err.exists() else "no error.txt"))
        all_done = all(st[s] == "COMPLETE" for s in ALL) and not [s for s in QUEUE if s not in pushed]
        if all_done and all(fetched.get(s) == "COMPLETE" for s in ALL):
            log("all kernels finished and fetched; exiting")
            return
        time.sleep(POLL)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # keep a trace for diagnosis
        import traceback
        log(f"scheduler crashed: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        sys.exit(1)
