@echo off
cd /d "D:\SL Project"
python -u scripts\03_run_tier1.py --tasks estrogen-alpha --workers 10 > results\local_tier1_log.txt 2>&1
python -u scripts\03_run_tier1.py --tasks freesolv estrogen-beta --workers 10 >> results\local_tier1_log.txt 2>&1
python -u scripts\03_run_tier1.py --tasks esol bbbp --workers 10 >> results\local_tier1_log.txt 2>&1
