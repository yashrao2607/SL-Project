"""Independent verification of the report (PRD 4.4 acceptance check).

1. Recomputes, directly from results/raw/tier1_runs.csv with plain pandas/scipy (not the molbench.stats
   code), every mean, paired Wilcoxon p-value, ablation contribution and generalization gap that
   reports/report_numbers.json tracks, and checks agreement.
2. Checks the charter's success criteria against the produced artefacts.
3. Writes reports/VERIFICATION.md with the outcome. Exits non-zero on any failure.
"""
from __future__ import annotations

import json
import sys
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats as ss

from molbench import config as C

TOL = 1e-3
failures, checks = [], []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    if not ok:
        failures.append(f"{name}: {detail}")


def primary(task):
    return C.PRIMARY_METRIC[C.TASKS[task]["type"]]


def main():
    R = C.RESULTS
    numbers = json.loads((C.REPORTS / "report_numbers.json").read_text())
    runs = pd.read_csv(R / "raw" / "tier1_runs.csv")
    runs = runs[runs["error"].isna() | (runs["error"].astype(str) == "")]
    models = sorted(runs["model"].unique())
    tasks = [t for t in C.TASK_NAMES if t in set(runs["task"])]

    # ---- 1. protocol completeness ----
    exp = len(C.TIER1_MODELS) * len(tasks) * 2 * 25
    check("tier1_run_count", len(runs) == exp, f"{len(runs)} rows, expected {exp}")
    check("tier1_no_nan_primary", all(np.isfinite(runs.apply(lambda r: r[primary(r['task'])], axis=1))), "NaN primary metric present")
    counts = runs.groupby(["model", "task", "split"]).size()
    check("every_cell_has_25_runs", bool((counts == 25).all()), f"min {counts.min()} max {counts.max()}")
    seeds = sorted(runs["seed"].unique().tolist())
    check("seeds_are_charter_seeds", seeds == C.SEEDS, str(seeds))
    check("folds_0_to_4", sorted(runs["fold"].unique().tolist()) == list(range(5)))
    # same folds for every model: identical n_test per (task, split, seed, fold)
    nt = runs.groupby(["task", "split", "seed", "fold"])["n_test"].nunique()
    check("identical_partitions_across_models", bool((nt == 1).all()))

    # ---- 2. recompute tracked numbers ----
    n_checked = 0
    for split in C.SPLITS:
        for m in models:
            for t in tasks:
                key = f"{split}_{m}_{t}_{primary(t)}_mean"
                if key in numbers:
                    x = runs[(runs["split"] == split) & (runs["model"] == m) & (runs["task"] == t)][primary(t)].values
                    check(key, abs(float(np.mean(x)) - numbers[key]) < TOL, f"recomputed {np.mean(x):.4f} vs report {numbers[key]}")
                    n_checked += 1
                dkey = f"diff_{split}_{m}_{t}"
                if dkey in numbers and m != "mat":
                    a = runs[(runs["split"] == split) & (runs["model"] == m) & (runs["task"] == t)].set_index(["seed", "fold"])[primary(t)]
                    b = runs[(runs["split"] == split) & (runs["model"] == "mat") & (runs["task"] == t)].set_index(["seed", "fold"])[primary(t)]
                    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
                    d = j["a"] - j["b"]
                    check(dkey, abs(float(d.mean()) - numbers[dkey]) < TOL, f"recomputed {d.mean():.4f} vs {numbers[dkey]}")
                    pkey = f"wilcoxon_p_{split}_{m}_{t}"
                    if pkey in numbers and len(j) >= 2 and not np.allclose(d, 0):
                        p = float(ss.wilcoxon(j["a"], j["b"]).pvalue)
                        check(pkey, abs(p - numbers[pkey]) < 1e-3, f"recomputed p={p:.4f} vs {numbers[pkey]}")
                    check(f"npairs_{split}_{m}_{t}", len(j) == 25, f"{len(j)} pairs")
                    n_checked += 2
        for t in tasks:
            for comp, abl_model in [("graph_structure", "mat_nograph"), ("3D_distances", "mat_nodistance"), ("self-attention", "mat_noattention")]:
                key = f"abl_{split}_{t}_{comp}"
                if key in numbers:
                    a = runs[(runs["split"] == split) & (runs["model"] == "mat") & (runs["task"] == t)].set_index(["seed", "fold"])[primary(t)]
                    b = runs[(runs["split"] == split) & (runs["model"] == abl_model) & (runs["task"] == t)].set_index(["seed", "fold"])[primary(t)]
                    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
                    contrib = float((j["a"] - j["b"]).mean())
                    check(key, abs(contrib - numbers[key]) < TOL, f"recomputed {contrib:.4f} vs {numbers[key]}")
                    n_checked += 1
    for m in models:
        for t in tasks:
            key = f"gap_{m}_{t}"
            if key in numbers:
                r = runs[(runs["split"] == "random") & (runs["model"] == m) & (runs["task"] == t)][primary(t)].mean()
                s = runs[(runs["split"] == "scaffold") & (runs["model"] == m) & (runs["task"] == t)][primary(t)].mean()
                check(key, abs(float(r - s) - numbers[key]) < TOL, f"recomputed {r - s:.4f} vs {numbers[key]}")
                n_checked += 1
    check("tracked_numbers_recomputed", n_checked > 0, f"{n_checked} quantities recomputed")

    # ---- 3. charter success criteria ----
    summ = pd.read_csv(R / "summary" / "summary_tier1.csv")
    check("clf_metrics_reported", set(C.CLF_METRICS) <= set(summ[summ["task_type"] == "clf"]["metric"]))
    check("reg_metrics_reported", set(C.REG_METRICS) <= set(summ[summ["task_type"] == "reg"]["metric"]))
    check("ci95_reported", {"ci95_low", "ci95_high"} <= set(summ.columns) and summ[["ci95_low", "ci95_high"]].notna().all().all())
    vs = pd.read_csv(R / "stats" / "vs_mat_tier1.csv")
    check("paired_t_and_wilcoxon", {"t_p", "wilcoxon_p"} <= set(vs.columns) and (vs["n_pairs"] == 25).all())
    check("pvalues_in_unit_interval", bool(((vs["t_p"].dropna() >= 0) & (vs["t_p"].dropna() <= 1)).all()))
    check("generalization_gap_reported", (R / "stats" / "generalization_gap_tier1.csv").exists())
    check("ablations_reported", (R / "stats" / "ablations_tier1.csv").exists())
    check("hybrid_evaluated", "hybrid" in models)
    check("gcn_rf_svm_evaluated", {"gcn", "rf", "svm"} <= set(models))
    check("pretrained_vs_scratch_reported", (R / "stats" / "pretrained_vs_scratch.csv").exists())
    check("attention_analysis_reported", len(list(C.RESULTS_ATTENTION.glob("*_head_stats.csv"))) > 0)
    check("figures_present", len(list(C.RESULTS_FIGURES.glob("*.png"))) >= 6, f"{len(list(C.RESULTS_FIGURES.glob('*.png')))} figures")
    report = (C.REPORTS / "REPORT.md").read_text(encoding="utf-8")
    for fig in C.RESULTS_FIGURES.glob("fig_*.png"):
        if fig.name in report:
            check(f"figure_linked_{fig.name}", True)
    check("report_exists", len(report) > 5000, f"{len(report)} chars")

    # ---- write VERIFICATION.md ----
    lines = ["# Verification report\n", f"Generated {date.today().isoformat()} by `scripts/verify_report.py`.\n",
             f"Checks run: {len(checks)}; failed: {len(failures)}.\n", "| Check | Result | Detail |", "|---|---|---|"]
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    (C.REPORTS / "VERIFICATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[verify] {len(checks)} checks, {len(failures)} failures")
    for f in failures:
        print("  FAIL:", f)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
