"""Phase 4.1: statistical evaluation of Tier-1 and Tier-2 runs.

Outputs (results/summary and results/stats):
  summary_tier1.csv            mean / std / 95% CI (t and bootstrap) per task, split, model, metric
  summary_tier2.csv            same for the pretrained-vs-scratch runs
  vs_mat_tier1.csv             paired t-test + Wilcoxon of every model against MAT (Holm-adjusted too)
  ablations_tier1.csv          contribution of graph / distance / attention terms (MAT minus ablation)
  generalization_gap_tier1.csv random-minus-scaffold gap per model and task
  pretrained_vs_scratch.csv    paired tests pretrained vs scratch (Tier 2)
  average_ranks_tier1.csv      MAT-paper-style average ranks per split
  regression_original_units.csv RMSE / MAE converted to log10(mol/L) and kcal/mol
"""
from __future__ import annotations

import json

import pandas as pd

from molbench import config as C
from molbench import stats as S


def original_units(summary: pd.DataFrame) -> pd.DataFrame:
    log = json.loads((C.DATA_PROCESSED / "cleaning_log.json").read_text())
    rows = []
    for task in C.REG_TASKS:
        info = log.get(task, {}).get("original_units")
        if not info:
            continue
        a = info["scale_a"]
        sub = summary[(summary["task"] == task) & (summary["metric"].isin(["rmse", "mae"]))]
        for _, r in sub.iterrows():
            rows.append({"task": task, "split": r["split"], "model": r["model"], "metric": r["metric"],
                         "unit": info["unit"], "mean_original_units": r["mean"] * a,
                         "ci95_low_original_units": r["ci95_low"] * a, "ci95_high_original_units": r["ci95_high"] * a,
                         "scale_a": a, "n_matched_source_molecules": info["n_matched"]})
    return pd.DataFrame(rows)


def main():
    C.ensure_dirs()
    t1_path = C.RESULTS_RAW / "tier1_runs.csv"
    t2_path = C.RESULTS_RAW / "tier2_runs.csv"
    if t1_path.exists():
        runs = S.load_runs(t1_path)
        print(f"[stats] tier1 rows: {len(runs)}")
        summ = S.summarize(runs)
        summ.to_csv(C.RESULTS_SUMMARY / "summary_tier1.csv", index=False)
        S.compare_to_reference(runs, reference="mat").to_csv(C.RESULTS_STATS / "vs_mat_tier1.csv", index=False)
        S.ablation_contributions(runs).to_csv(C.RESULTS_STATS / "ablations_tier1.csv", index=False)
        S.generalization_gap(runs).to_csv(C.RESULTS_STATS / "generalization_gap_tier1.csv", index=False)
        S.average_ranks(summ).to_csv(C.RESULTS_STATS / "average_ranks_tier1.csv", index=False)
        original_units(summ).to_csv(C.RESULTS_SUMMARY / "regression_original_units.csv", index=False)
        # Hybrid vs MAT, GCN vs MAT etc. are already in vs_mat; also RF as reference for the classical view.
        S.compare_to_reference(runs, reference="rf").to_csv(C.RESULTS_STATS / "vs_rf_tier1.csv", index=False)
        counts = runs.groupby(["model", "split"]).size().unstack(fill_value=0)
        print(counts)
    if t2_path.exists():
        runs2 = S.load_runs(t2_path)
        print(f"[stats] tier2 rows: {len(runs2)}")
        summ2 = S.summarize(runs2)
        summ2.to_csv(C.RESULTS_SUMMARY / "summary_tier2.csv", index=False)
        S.compare_to_reference(runs2, reference="mat_large_scratch", models=["mat_large_pretrained"]).to_csv(
            C.RESULTS_STATS / "pretrained_vs_scratch.csv", index=False)
        S.generalization_gap(runs2).to_csv(C.RESULTS_STATS / "generalization_gap_tier2.csv", index=False)
        if t1_path.exists():
            # Small scratch MAT vs large pretrained MAT on the shared (seed, fold 0) runs.
            small = runs[(runs["model"] == "mat") & (runs["fold"] == 0)]
            both = pd.concat([small, runs2], ignore_index=True)
            S.compare_to_reference(both, reference="mat", models=C.TIER2_MODELS).to_csv(
                C.RESULTS_STATS / "large_vs_small_mat_fold0.csv", index=False)
    print("[stats] done")


if __name__ == "__main__":
    main()
