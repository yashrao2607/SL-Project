"""Phase 4.4: build reports/REPORT.md and reports/report_numbers.json from the results CSVs.

Every number in the report is read from results/ (never typed by hand); scripts/verify_report.py
recomputes the key quantities independently from the raw run CSVs and checks them.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from molbench import config as C

NUMBERS = {}          # name -> value, dumped to report_numbers.json for verification
LBL = C.MODEL_LABELS
TASK_LABELS = {"bbbp": "BBBP", "esol": "ESOL", "freesolv": "FreeSolv", "estrogen-alpha": "Estrogen-α",
               "estrogen-beta": "Estrogen-β", "metstab-high": "MetStab-high", "metstab-low": "MetStab-low"}
METRIC_LABELS = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1", "rmse": "RMSE", "mae": "MAE", "r2": "R²"}


def num(name, value, digits=3):
    v = float(value)
    NUMBERS[name] = round(v, digits)
    return f"{v:.{digits}f}"


def pval(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def stars(p):
    if p is None or np.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def primary(task):
    return C.PRIMARY_METRIC[C.TASKS[task]["type"]]


def main():
    C.ensure_dirs()
    R = C.RESULTS
    log = json.loads((C.DATA_PROCESSED / "cleaning_log.json").read_text())
    runs1 = pd.read_csv(R / "raw" / "tier1_runs.csv")
    runs1 = runs1[runs1["error"].isna() | (runs1["error"].astype(str) == "")]
    summ = pd.read_csv(R / "summary" / "summary_tier1.csv")
    vs_mat = pd.read_csv(R / "stats" / "vs_mat_tier1.csv")
    vs_mat_seed = pd.read_csv(R / "stats" / "vs_mat_tier1_seedlevel.csv")
    abl = pd.read_csv(R / "stats" / "ablations_tier1.csv")
    abl_seed = pd.read_csv(R / "stats" / "ablations_tier1_seedlevel.csv")
    gap = pd.read_csv(R / "stats" / "generalization_gap_tier1.csv")
    ranks = pd.read_csv(R / "stats" / "average_ranks_tier1.csv")
    orig = pd.read_csv(R / "summary" / "regression_original_units.csv")
    best = pd.read_csv(R / "tuning" / "best_configs.csv")
    tier2 = (R / "raw" / "tier2_runs.csv").exists()
    if tier2:
        runs2 = pd.read_csv(R / "raw" / "tier2_runs.csv")
        runs2 = runs2[runs2["error"].isna() | (runs2["error"].astype(str) == "")]
        summ2 = pd.read_csv(R / "summary" / "summary_tier2.csv")
        pvs = pd.read_csv(R / "stats" / "pretrained_vs_scratch.csv")
        lvs = pd.read_csv(R / "stats" / "large_vs_small_mat_fold0.csv")
    tasks = [t for t in C.TASK_NAMES if t in set(summ["task"])]
    models = [m for m in C.TIER1_MODELS if m in set(summ["model"])]

    L = []
    L.append("# Statistical Evaluation and Extension of the Molecule Attention Transformer (MAT)\n")
    L.append("DSC4804 Statistical Learning, Group 14 (Yash Yadav, Kartavya Dev, Akshat Nagori). "
             f"Report generated {date.today().isoformat()} by `scripts/08_report.py` from the files in `results/`.\n")
    L.append("Every number below is produced by the pipeline; `scripts/verify_report.py` recomputes the key "
             "quantities from the raw per-run CSVs and checks them against this report.\n")

    # ------------------------------------------------------------------ 1. data
    L.append("## 1. Data\n")
    L.append("Datasets are the seven benchmark tasks released with the MAT paper (Maziarka et al., 2020; "
             "`ardigen/MAT` repository). Cleaning: largest fragment, canonical SMILES, duplicate removal "
             "(conflicting labels dropped), molecules with more than 100 heavy atoms removed. "
             "Scaffolds are Bemis-Murcko scaffolds without chirality.\n")
    rows = []
    for t in tasks:
        lg = log[t]
        extra = f"positives {100 * lg['positive_rate']:.1f} %" if C.TASKS[t]["type"] == "clf" else "z-scored labels"
        rows.append([TASK_LABELS[t], C.TASKS[t]["type"], lg["n_raw"], lg["n_clean"], lg["n_duplicate_rows_removed"],
                     lg["n_conflicting_label_molecules_dropped"], lg["n_too_large"], lg["n_scaffolds"], extra])
        NUMBERS[f"n_clean_{t}"] = lg["n_clean"]
    L.append(md_table(["Task", "Type", "Raw", "Clean", "Duplicate rows removed", "Conflicting molecules dropped",
                       ">100 heavy atoms", "Scaffolds", "Label"], rows) + "\n")
    fz = log["_featurization"]
    NUMBERS["n_unique_molecules"] = fz["n_unique_molecules"]
    NUMBERS["n_etkdg"] = fz["conformer_methods"].get("etkdg", 0)
    L.append(f"3D geometry: {fz['n_unique_molecules']} unique molecules; ETKDG (RDKit ETKDGv3 + UFF relaxation) "
             f"succeeded for {fz['conformer_methods'].get('etkdg', 0)} and the original code's 2D fallback was used for "
             f"{fz['conformer_methods'].get('2d', 0)}. Atom features: 28-dimensional (dummy-node indicator + one-hot "
             "atomic number, degree, H count, formal charge, ring flag, aromatic flag), identical to the MAT authors' "
             "featuriser. Fingerprints: ECFP4 (Morgan radius 2, 2048 bits).\n")
    for t in ["esol", "freesolv"]:
        if t in log and "original_units" in log[t]:
            ou = log[t]["original_units"]
            L.append(f"- {TASK_LABELS[t]} labels in the MAT release are z-scored; matching {ou['n_matched']} molecules to the "
                     f"source ({ou['source']}) recovers y = {ou['scale_a']:.4f}·z + ({ou['offset_b']:.4f}) in {ou['unit']} "
                     f"(max residual {ou['max_abs_residual']:.1e}), so RMSE/MAE in original units are {ou['scale_a']:.4f} × the z-scored values.")
    L.append("")

    # ------------------------------------------------------------------ 2. protocol
    L.append("## 2. Protocol\n")
    n_runs = len(runs1)
    NUMBERS["n_tier1_runs"] = n_runs
    L.append(f"- Splits: Random (stratified 5-fold) and Scaffold (balanced Bemis-Murcko 5-fold, test scaffolds unseen in "
             "training) for seeds 42, 123, 456, 789, 1011, i.e. 25 train/validation/test partitions per split type "
             "(72 % / 8 % / 20 %). All models share the same partitions, so every comparison is paired.\n"
             "- Tier 1 models (full protocol): RF (ECFP4), SVM/SVR (Tanimoto kernel on ECFP4), GCN (PyTorch Geometric), "
             "MAT (from scratch), MAT-NoGraph, MAT-NoDistance, MAT-NoAttention, ECFP+MAT late-fusion hybrid. "
             f"Completed runs: {n_runs} (expected {len(models) * len(tasks) * 2 * 25}).\n"
             "- Hyperparameters: grid search on the seed-42 / fold-0 validation set per (family, task, split); the winner is "
             "frozen for all 25 runs. Ablations and the hybrid inherit MAT's configuration.\n"
             "- Deep models: Adam, gradient clipping 5.0, early stopping on validation ROC-AUC / RMSE "
             f"(max {int(runs1['max_epochs'].max())} epochs, patience {int(runs1['patience'].max())}), best validation checkpoint restored.\n"
             "- Metrics: ROC-AUC, PR-AUC, F1@0.5 (classification); RMSE, MAE, R² (regression). Primary metric for model "
             "selection and ranking: ROC-AUC / RMSE.\n"
             "- Statistics: mean ± 95 % CI (t-distribution over 25 fold scores), paired t-test and Wilcoxon signed-rank test "
             "against MAT on the 25 paired folds, Holm correction within each (task, split, metric) family, and a "
             "seed-level robustness check (5 seed means). Significance threshold p < 0.05.\n")
    L.append("Frozen configurations (selected on seed 42 / fold 0 validation; ablations and the hybrid use the MAT row):\n")
    rows = []
    for fam in ["rf", "svm", "gcn", "mat"]:
        for t in tasks:
            sub = best[(best["family"] == fam) & (best["task"] == t)].set_index("split")
            if sub.empty:
                continue
            rows.append([fam, TASK_LABELS[t], sub.loc["random", "config"] if "random" in sub.index else "–",
                         sub.loc["scaffold", "config"] if "scaffold" in sub.index else "–"])
    L.append(md_table(["Family", "Task", "Random split", "Scaffold split"], rows) + "\n")

    # ------------------------------------------------------------------ 3. main results
    L.append("## 3. Main results (Tier 1)\n")
    for split in C.SPLITS:
        L.append(f"### 3.{1 if split == 'random' else 2} {split.capitalize()} split, primary metric (mean ± 95 % CI over 25 folds)\n")
        L.append(f"![]( ../results/figures/fig_performance_{split}.png)\n")
        header = ["Model"] + [f"{TASK_LABELS[t]} ({METRIC_LABELS[primary(t)]}{'↓' if primary(t) == 'rmse' else '↑'})" for t in tasks]
        rows = []
        for m in models:
            cells = [LBL[m]]
            for t in tasks:
                r = summ[(summ["split"] == split) & (summ["model"] == m) & (summ["task"] == t) & (summ["metric"] == primary(t))]
                if r.empty:
                    cells.append("–")
                    continue
                r = r.iloc[0]
                key = split + "_" + m + "_" + t + "_" + primary(t) + "_mean"
                cells.append(f"{num(key, r['mean'])} ± {(r['ci95_high'] - r['ci95_low']) / 2:.3f}")
            rows.append(cells)
        L.append(md_table(header, rows) + "\n")
        rk = ranks[ranks["split"] == split].sort_values("avg_rank")
        L.append("Average rank across the 7 tasks (1 = best): " + ", ".join(
            f"{LBL[r['model']]} {num('rank_' + split + '_' + r['model'], r['avg_rank'], 2)}" for _, r in rk.iterrows()) + ".\n")
    L.append("![]( ../results/figures/fig_average_ranks.png)\n")
    L.append("Regression metrics converted to original units (RMSE and MAE scale linearly with the recovered factor):\n")
    rows = []
    for _, r in orig[orig["metric"] == "rmse"].sort_values(["task", "split", "model"]).iterrows():
        rows.append([TASK_LABELS[r["task"]], r["split"], LBL[r["model"]], f"{r['mean_original_units']:.3f} {r['unit']}",
                     f"[{r['ci95_low_original_units']:.3f}, {r['ci95_high_original_units']:.3f}]"])
    L.append(md_table(["Task", "Split", "Model", "RMSE (original units)", "95 % CI"], rows) + "\n")

    # ------------------------------------------------------------------ 4. tests vs MAT
    L.append("## 4. Statistical comparison against MAT\n")
    L.append("Difference = model − MAT on the primary metric over the 25 paired folds. `t` = paired t-test p-value, "
             "`W` = Wilcoxon signed-rank p-value, `Holm` = Holm-adjusted Wilcoxon p within the (task, split) family; "
             "stars: * p<0.05, ** p<0.01, *** p<0.001 (Wilcoxon). Positive differences favour the model for ROC-AUC and "
             "negative differences favour the model for RMSE. Seed-level column: Wilcoxon p on the 5 seed means.\n")
    for split in C.SPLITS:
        L.append(f"### 4.{1 if split == 'random' else 2} {split.capitalize()} split\n")
        rows = []
        for t in tasks:
            for m in [x for x in models if x != "mat"]:
                r = vs_mat[(vs_mat["split"] == split) & (vs_mat["task"] == t) & (vs_mat["model"] == m) & (vs_mat["metric"] == primary(t))]
                rs = vs_mat_seed[(vs_mat_seed["split"] == split) & (vs_mat_seed["task"] == t) & (vs_mat_seed["model"] == m) & (vs_mat_seed["metric"] == primary(t))]
                if r.empty:
                    continue
                r = r.iloc[0]
                key = f"diff_{split}_{m}_{t}"
                rows.append([TASK_LABELS[t], LBL[m], METRIC_LABELS[primary(t)], num(key, r["mean_diff"]),
                             f"[{r['diff_ci95_low']:.3f}, {r['diff_ci95_high']:.3f}]", pval(r["t_p"]),
                             pval(r["wilcoxon_p"]) + stars(r["wilcoxon_p"]), pval(r["wilcoxon_p_holm"]),
                             pval(rs.iloc[0]["wilcoxon_p"]) if not rs.empty else "n/a",
                             "MAT" if r["reference_better"] else LBL[m]])
                NUMBERS[f"wilcoxon_p_{split}_{m}_{t}"] = round(float(r["wilcoxon_p"]), 4)
        L.append(md_table(["Task", "Model", "Metric", "Δ (model − MAT)", "95 % CI of Δ", "t p", "W p", "Holm p",
                           "seed-level W p", "Better"], rows) + "\n")
        # summary counts
        for m in [x for x in models if x != "mat"]:
            sub = vs_mat[(vs_mat["split"] == split) & (vs_mat["model"] == m)]
            sub = sub[sub.apply(lambda q: q["metric"] == primary(q["task"]), axis=1)]
            sig = sub[sub["wilcoxon_p"] < 0.05]
            mat_wins = int((sig["reference_better"]).sum())
            model_wins = int((~sig["reference_better"]).sum())
            NUMBERS[f"sigcount_{split}_{m}_matwins"] = mat_wins
            NUMBERS[f"sigcount_{split}_{m}_modelwins"] = model_wins
            L.append(f"- {LBL[m]} vs MAT ({split}): MAT significantly better on {mat_wins}/{len(sub)} tasks, "
                     f"{LBL[m]} significantly better on {model_wins}/{len(sub)} tasks (Wilcoxon p < 0.05).")
        L.append("")

    # ------------------------------------------------------------------ 5. ablations
    L.append("## 5. Ablation study: contribution of each MAT term\n")
    L.append("Contribution = MAT − ablation on the primary metric (paired, 25 folds). For ROC-AUC a positive value means "
             "the term helps; for RMSE a negative value means the term helps (lower error). Removing a term "
             "re-distributes its λ weight equally to the remaining two terms.\n")
    L.append("![]( ../results/figures/fig_ablations.png)\n")
    rows = []
    for split in C.SPLITS:
        for t in tasks:
            for comp in ["graph structure", "3D distances", "self-attention"]:
                r = abl[(abl["split"] == split) & (abl["task"] == t) & (abl["component"] == comp) & (abl["metric"] == primary(t))]
                rs = abl_seed[(abl_seed["split"] == split) & (abl_seed["task"] == t) & (abl_seed["component"] == comp) & (abl_seed["metric"] == primary(t))]
                if r.empty:
                    continue
                r = r.iloc[0]
                helps = (r["contribution"] > 0) if C.HIGHER_IS_BETTER[primary(t)] else (r["contribution"] < 0)
                rows.append([split, TASK_LABELS[t], comp, METRIC_LABELS[primary(t)],
                             num("abl_" + split + "_" + t + "_" + comp.replace(" ", "_"), r["contribution"]),
                             f"[{r['contribution_ci95_low']:.3f}, {r['contribution_ci95_high']:.3f}]", pval(r["t_p"]),
                             pval(r["wilcoxon_p"]) + stars(r["wilcoxon_p"]),
                             pval(rs.iloc[0]["wilcoxon_p"]) if not rs.empty else "n/a",
                             ("helps" if helps else "hurts") + (" (sig.)" if r["wilcoxon_p"] < 0.05 else " (n.s.)")])
    L.append(md_table(["Split", "Task", "Term removed", "Metric", "Contribution", "95 % CI", "t p", "W p", "seed-level W p",
                       "Verdict"], rows) + "\n")
    for comp in ["graph structure", "3D distances", "self-attention"]:
        for split in C.SPLITS:
            sub = abl[(abl["split"] == split) & (abl["component"] == comp)]
            sub = sub[sub.apply(lambda q: q["metric"] == primary(q["task"]), axis=1)]
            helps = sub.apply(lambda q: (q["contribution"] > 0) if C.HIGHER_IS_BETTER[q["metric"]] else (q["contribution"] < 0), axis=1)
            sig = sub["wilcoxon_p"] < 0.05
            ckey = comp.replace(" ", "_")
            NUMBERS["abl_sig_helps_" + split + "_" + ckey] = int((helps & sig).sum())
            NUMBERS["abl_sig_hurts_" + split + "_" + ckey] = int((~helps & sig).sum())
            L.append(f"- {comp} ({split}): helps significantly on {int((helps & sig).sum())}/{len(sub)} tasks, "
                     f"hurts significantly on {int((~helps & sig).sum())}/{len(sub)} tasks, point estimate helps on {int(helps.sum())}/{len(sub)}.")
    L.append("")

    # ------------------------------------------------------------------ 6. generalization gap
    L.append("## 6. Generalization gap (random − scaffold)\n")
    L.append("Gap = mean random-split score − mean scaffold-split score on the primary metric (Welch 95 % CI; the two "
             "split types are independent partitions). A positive ROC-AUC gap or a negative RMSE gap means the model "
             "performs worse on unseen scaffolds.\n")
    L.append("![]( ../results/figures/fig_generalization_gap.png)\n")
    header = ["Model"] + [f"{TASK_LABELS[t]} ({METRIC_LABELS[primary(t)]})" for t in tasks]
    rows = []
    for m in models:
        cells = [LBL[m]]
        for t in tasks:
            r = gap[(gap["model"] == m) & (gap["task"] == t) & (gap["metric"] == primary(t))]
            if r.empty:
                cells.append("–")
                continue
            r = r.iloc[0]
            cells.append(f"{num('gap_' + m + '_' + t, r['gap'], 3)} [{r['gap_ci95_low']:.3f}, {r['gap_ci95_high']:.3f}]")
        rows.append(cells)
    L.append(md_table(header, rows) + "\n")
    g_prim = gap[gap.apply(lambda q: q["metric"] == primary(q["task"]), axis=1)].copy()
    g_prim["gap_signed"] = g_prim.apply(lambda q: q["gap"] if C.HIGHER_IS_BETTER[q["metric"]] else -q["gap"], axis=1)
    clf_gap = g_prim[g_prim["task_type"] == "clf"].groupby("model")["gap_signed"].mean()
    L.append("Mean ROC-AUC drop from random to scaffold split over the five classification tasks: " + ", ".join(
        f"{LBL[m]} {num('meangap_clf_' + m, clf_gap[m], 3)}" for m in models if m in clf_gap) + ".\n")

    # ------------------------------------------------------------------ 7. pretrained vs scratch
    if tier2:
        L.append("## 7. Pretrained MAT versus scratch MAT (Tier 2, released architecture: 8 layers, d_model 1024, 16 heads, 42 M parameters)\n")
        n2 = len(runs2)
        NUMBERS["n_tier2_runs"] = n2
        L.append("Note on the tests: with n = 5 paired seeds the two-sided Wilcoxon signed-rank test cannot go below p = 0.0625, "
                 "so it can never reach the 0.05 threshold in this section; the paired t-test is the informative test here.\n")
        L.append(f"Protocol: fold 0 of every seed, both split types, {int(runs2['max_epochs'].max())} epochs maximum with early stopping "
                 f"(patience {int(runs2['patience'].max())}), identical for both variants; {n2} runs. The pretrained variant "
                 "loads the authors' released checkpoint (all encoder weights, the pretraining head is discarded); the "
                 "scratch variant uses the same architecture with Xavier initialisation.\n")
        L.append("![]( ../results/figures/fig_pretrained_vs_scratch.png)\n")
        rows = []
        for split in C.SPLITS:
            for t in tasks:
                pr = summ2[(summ2["split"] == split) & (summ2["task"] == t) & (summ2["metric"] == primary(t))].set_index("model")
                cmp_ = pvs[(pvs["split"] == split) & (pvs["task"] == t) & (pvs["metric"] == primary(t))]
                small = summ[(summ["split"] == split) & (summ["task"] == t) & (summ["metric"] == primary(t)) & (summ["model"] == "mat")]
                if pr.empty or cmp_.empty:
                    continue
                c = cmp_.iloc[0]
                rows.append([split, TASK_LABELS[t], METRIC_LABELS[primary(t)],
                             num(f"t2_{split}_{t}_pretrained", pr.loc["mat_large_pretrained", "mean"]),
                             num(f"t2_{split}_{t}_scratch", pr.loc["mat_large_scratch", "mean"]),
                             f"{small.iloc[0]['mean']:.3f}" if not small.empty else "–",
                             num(f"t2_{split}_{t}_diff", c["mean_diff"]), pval(c["t_p"]), pval(c["wilcoxon_p"]) + stars(c["wilcoxon_p"]),
                             "pretrained" if not c["reference_better"] else "scratch"])
        n2s = int(summ2["n"].max())
        L.append(md_table(["Split", "Task", "Metric", f"Pretrained (n={n2s})", f"Scratch-large (n={n2s})", "Small MAT (25 folds)",
                           "Δ (pre − scratch)", "t p", "W p", "Better"], rows) + "\n")
        for split in C.SPLITS:
            sub = pvs[pvs["split"] == split]
            sub = sub[sub.apply(lambda q: q["metric"] == primary(q["task"]), axis=1)]
            wins = int((~sub["reference_better"]).sum())
            sig_t = int(((~sub["reference_better"]) & (sub["t_p"] < 0.05)).sum())
            sig_w = int(((~sub["reference_better"]) & (sub["wilcoxon_p"] < 0.05)).sum())
            NUMBERS[f"t2_pre_wins_{split}"] = wins
            NUMBERS[f"t2_pre_sig_t_{split}"] = sig_t
            NUMBERS[f"t2_pre_sig_{split}"] = sig_w
            L.append(f"- {split}: pretraining improves the point estimate on {wins}/{len(sub)} tasks"
                     + (f", significantly on {sig_t}/{len(sub)} by the paired t-test and {sig_w}/{len(sub)} by Wilcoxon (n = {n2s} seeds)."
                        if n2s >= 2 else
                        f" (single seed locally; significance requires the 5-seed GPU protocol in notebooks/colab_tier2.ipynb)."))
        L.append("\nLarge models versus the small scratch MAT of Tier 1 on the same fold-0 partitions (paired over 5 seeds):\n")
        rows = []
        for _, r in lvs[lvs.apply(lambda q: q["metric"] == primary(q["task"]), axis=1)].iterrows():
            rows.append([r["split"], TASK_LABELS[r["task"]], LBL[r["model"]], METRIC_LABELS[r["metric"]], f"{r['mean_diff']:+.3f}",
                         pval(r["wilcoxon_p"]) + stars(r["wilcoxon_p"]), "small MAT" if r["reference_better"] else LBL[r["model"]]])
        L.append(md_table(["Split", "Task", "Large model", "Metric", "Δ (large − small MAT)", "W p", "Better"], rows) + "\n")

    # ------------------------------------------------------------------ 8. hybrid
    L.append("## 8. ECFP + MAT hybrid\n")
    rows = []
    for split in C.SPLITS:
        for t in tasks:
            r = vs_mat[(vs_mat["split"] == split) & (vs_mat["task"] == t) & (vs_mat["model"] == "hybrid") & (vs_mat["metric"] == primary(t))]
            rf = vs_mat[(vs_mat["split"] == split) & (vs_mat["task"] == t) & (vs_mat["model"] == "rf") & (vs_mat["metric"] == primary(t))]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([split, TASK_LABELS[t], METRIC_LABELS[primary(t)], f"{r['reference_mean']:.3f}", f"{r['model_mean']:.3f}",
                         f"{rf.iloc[0]['model_mean']:.3f}" if not rf.empty else "–", f"{r['mean_diff']:+.3f}",
                         pval(r["wilcoxon_p"]) + stars(r["wilcoxon_p"]), "hybrid" if not r["reference_better"] else "MAT"])
    L.append(md_table(["Split", "Task", "Metric", "MAT", "Hybrid", "RF", "Δ (hybrid − MAT)", "W p", "Better"], rows) + "\n")
    small_tasks = sorted(tasks, key=lambda t: log[t]["n_clean"])[:2]
    L.append(f"Smallest datasets (low-data regime): {', '.join(TASK_LABELS[t] for t in small_tasks)}.\n")

    # ------------------------------------------------------------------ 9. attention
    att_files = sorted(C.RESULTS_ATTENTION.glob("*_head_stats.csv"))
    if att_files:
        L.append("## 9. Attention analysis\n")
        L.append("Trained MAT (seed 42, fold 0) on the test molecules. `self_attn` is the learned softmax(QKᵀ) term, "
                 "`attn` the fused matrix actually used (λ-weighted sum with the distance and adjacency terms). "
                 "Bonded share = attention mass an atom sends to its bonded neighbours; Spearman(attention, 1/distance) "
                 "measures how much the learned attention already tracks 3D proximity; entropy is the mean row entropy "
                 "(uniform attention over n atoms would give ln n). The uniform baseline is what a molecule-size-matched "
                 "uniform attention would give.\n")
        rows = []
        for hs in att_files:
            task, split = hs.name.replace("_head_stats.csv", "").rsplit("_", 1)
            st = pd.read_csv(hs)
            base = json.loads((C.RESULTS_ATTENTION / f"{task}_{split}_uniform_baseline.json").read_text())
            for kind in ["self_attn", "attn"]:
                s = st[st["kind"] == kind]
                rows.append([TASK_LABELS[task], split, kind, num(f"att_{task}_{split}_{kind}_bonded", s["bonded_share"].mean()),
                             f"{base['bonded_share']:.3f}", num(f"att_{task}_{split}_{kind}_self", s["self_share"].mean()),
                             num(f"att_{task}_{split}_{kind}_dummy", s["dummy_share"].mean()),
                             num(f"att_{task}_{split}_{kind}_distcorr", s["dist_corr_spearman"].mean()),
                             num(f"att_{task}_{split}_{kind}_entropy", s["entropy"].mean(), 2),
                             f"{s['bonded_share'].min():.3f} – {s['bonded_share'].max():.3f}"])
            L.append(f"![]( ../results/figures/fig_attention_heads_{task}_{split}.png)\n")
            L.append(f"![]( ../results/figures/fig_attention_classes_{task}_{split}.png)\n")
            L.append(f"![]( ../results/figures/fig_attention_examples_{task}_{split}.png)\n")
        L.append(md_table(["Task", "Split", "Kind", "Bonded share", "Uniform bonded", "Self share", "Dummy share",
                           "Spearman(att, 1/d)", "Entropy", "Bonded share range over heads"], rows) + "\n")
        rows = []
        for hs in att_files:
            task, split = hs.name.replace("_head_stats.csv", "").rsplit("_", 1)
            rec = pd.read_csv(C.RESULTS_ATTENTION / f"{task}_{split}_received_by_class.csv")
            last = rec[rec["layer"] == rec["layer"].max()]
            for _, r in last.iterrows():
                rows.append([TASK_LABELS[task], split, r["atom_class"], f"{r['mean_attention_received']:.4f}"])
        L.append("Mean self-attention received per atom by atom class (last layer):\n")
        L.append(md_table(["Task", "Split", "Atom class", "Mean attention received"], rows) + "\n")

    # ------------------------------------------------------------------ 10. verdict
    L.append("## 10. Verdict on the central hypothesis\n")
    L.append("The hypothesis is that adding graph structure and 3D distances to self-attention improves accuracy and "
             "out-of-distribution generalization. The evidence is summarised from Sections 4 to 6 (numbers are computed, "
             "not interpreted, here; the discussion in `docs/PROJECT_OVERVIEW.md` interprets them):\n")
    for comp in ["graph structure", "3D distances", "self-attention"]:
        for split in C.SPLITS:
            L.append(f"- Removing **{comp}** ({split} split) hurts significantly on "
                     f"{NUMBERS['abl_sig_helps_' + split + '_' + comp.replace(' ', '_')]}/7 tasks and helps significantly on "
                     f"{NUMBERS['abl_sig_hurts_' + split + '_' + comp.replace(' ', '_')]}/7 tasks.")
    for split in C.SPLITS:
        for m in ["rf", "svm", "gcn", "hybrid"]:
            if m in models:
                L.append(f"- {LBL[m]} vs MAT ({split}): MAT significantly better on {NUMBERS['sigcount_' + split + '_' + m + '_matwins']}/7, "
                         f"{LBL[m]} significantly better on {NUMBERS['sigcount_' + split + '_' + m + '_modelwins']}/7.")
    L.append("")
    L.append("## 11. Limitations\n")
    L.append("- CPU-only execution: the small MAT configurations (d_model 64–128, 2–4 layers) are far smaller than the "
             "paper's tuned models; the pretrained-vs-scratch comparison uses a reduced protocol (fold 0 of each seed) and a short epoch budget.\n"
             "- Scaffold K-fold: for ESOL and FreeSolv the benzene scaffold group is larger than one fold, so one scaffold "
             "test fold is identical across seeds; the seed-level tests are provided as the conservative check.\n"
             "- Fold scores within a cross-validation are not independent (overlapping training sets), so paired-test "
             "p-values are somewhat optimistic; Holm-adjusted and seed-level p-values are reported alongside.\n"
             "- ESOL/FreeSolv labels are z-scored in the released data; original-unit RMSE/MAE are recovered exactly through a linear map.\n")
    L.append("## 12. Reproducibility\n")
    wall = runs1["wall_time_s"].sum() / 3600
    NUMBERS["tier1_cpu_hours"] = round(float(wall), 2)
    dev = runs1["device"].fillna("cpu").value_counts().to_dict() if "device" in runs1 else {}
    dev_txt = ", ".join(f"{v} on {k}" for k, v in dev.items())
    L.append(f"- Tier 1: {n_runs} runs ({dev_txt}), {wall:.1f} hours of model fitting summed over runs (executed in parallel on the "
             "laptop's 14 CPU workers and on Kaggle T4 GPUs; the `device` column of `results/raw/tier1_runs.csv` records where each run executed).\n"
             "- Environment: Python 3.11.9 on Windows 11 (laptop, CPU) and the Kaggle Python image with CUDA (T4 GPUs); "
             "exact package versions used locally are pinned in `requirements.txt`; the `torch_version` column of the run CSVs records the GPU side.\n"
             "- Re-run everything with `python run_all.py`; individual steps are the numbered scripts in `scripts/`.\n")

    (C.REPORTS / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    (C.REPORTS / "report_numbers.json").write_text(json.dumps(NUMBERS, indent=1))
    print(f"[report] wrote reports/REPORT.md ({len(L)} blocks) and {len(NUMBERS)} tracked numbers")


if __name__ == "__main__":
    main()
