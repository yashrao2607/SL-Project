#!/usr/bin/env bash
# Final assembly once all runs exist: merge Kaggle outputs, statistics, attention, figures, report, verification.
set -e
cd "$(dirname "$0")/.."
echo "=== [$(date)] fetch/merge Kaggle results ===" && python -u scripts/fetch_kaggle_results.py "$@"
echo "=== [$(date)] 05_stats ==="      && python -u scripts/05_stats.py
echo "=== [$(date)] 06_attention ==="  && python -u scripts/06_attention.py
echo "=== [$(date)] 07_figures ==="    && python -u scripts/07_figures.py
echo "=== [$(date)] 08_report ==="     && python -u scripts/08_report.py
echo "=== [$(date)] verify_report ===" && python -u scripts/verify_report.py
echo "=== [$(date)] FINALIZE COMPLETE ==="
