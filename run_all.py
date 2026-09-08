"""One-command reproduction of the whole study.

    python run_all.py            # full protocol (hours on CPU; see README for timings)
    python run_all.py --smoke    # tiny end-to-end check (one task, one seed, one fold, few epochs)

Each step is resume-safe: completed runs are skipped, so the command can be re-run after an
interruption.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd: list[str]):
    print("\n$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"step failed: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--skip-tier2", action="store_true", help="skip the 42M-parameter pretrained-vs-scratch runs")
    args = ap.parse_args()
    run([PY, "-m", "pytest", "tests", "-q"])
    run([PY, "scripts/01_prepare_data.py", "--jobs", str(max(1, args.workers))])
    if args.smoke:
        run([PY, "scripts/02_tune.py", "--families", "rf", "svm", "gcn", "mat", "--tasks", "freesolv",
             "--workers", str(args.workers), "--threads", str(args.threads), "--max-epochs", "3", "--patience", "2"])
        run([PY, "scripts/03_run_tier1.py", "--tasks", "freesolv", "--workers", str(args.workers),
             "--threads", str(args.threads), "--max-epochs", "3", "--patience", "2"])
        run([PY, "scripts/04_run_tier2.py", "--tasks", "freesolv", "--seeds", "42", "--workers", "1",
             "--threads", "8", "--max-epochs", "1", "--patience", "1"])
        run([PY, "scripts/05_stats.py"])
        run([PY, "scripts/06_attention.py", "--tasks", "freesolv", "--splits", "random"])
        run([PY, "scripts/07_figures.py"])
        return
    run([PY, "scripts/02_tune.py", "--workers", str(args.workers), "--threads", str(args.threads)])
    run([PY, "scripts/03_run_tier1.py", "--workers", str(args.workers), "--threads", str(args.threads)])
    if not args.skip_tier2:
        run([PY, "scripts/04_run_tier2.py", "--local"])
    run([PY, "scripts/05_stats.py"])
    run([PY, "scripts/06_attention.py"])
    run([PY, "scripts/07_figures.py"])
    run([PY, "scripts/08_report.py"])
    run([PY, "scripts/verify_report.py"])


if __name__ == "__main__":
    main()
