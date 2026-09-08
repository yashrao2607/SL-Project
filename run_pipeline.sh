#!/usr/bin/env bash
# Full pipeline after data preparation (resume-safe; re-run to continue after an interruption).
set -e
cd "$(dirname "$0")"
echo "=== [$(date)] 02_tune ==="       && python -u scripts/02_tune.py
echo "=== [$(date)] 03_run_tier1 ==="  && python -u scripts/03_run_tier1.py
echo "=== [$(date)] 04_run_tier2 (local CPU protocol) ===" && python -u scripts/04_run_tier2.py --local
echo "=== [$(date)] 05_stats ==="      && python -u scripts/05_stats.py
echo "=== [$(date)] 06_attention ==="  && python -u scripts/06_attention.py
echo "=== [$(date)] 07_figures ==="    && python -u scripts/07_figures.py
echo "=== [$(date)] 08_report ==="     && python -u scripts/08_report.py
echo "=== [$(date)] verify_report ===" && python -u scripts/verify_report.py
echo "=== [$(date)] PIPELINE COMPLETE ==="
