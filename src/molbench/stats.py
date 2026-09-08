"""Statistical evaluation (PRD section 3.4): summaries with 95% CIs, paired tests, Holm correction,
generalization gap, ablation contributions and average ranks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as ss

from . import config as C

PAIR_KEYS = ["seed", "fold"]


def _metrics_for(task_type: str):
    return C.CLF_METRICS if task_type == "clf" else C.REG_METRICS


def t_ci(x: np.ndarray, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
    h = ss.t.ppf(1 - alpha / 2, n - 1) * se
    return (m - h, m + h)


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.RandomState(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))))


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (NaN preserved)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * p[i])
        running = max(running, adj)
        out[i] = running
    return out


# --------------------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------------------
def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, split, model), g in runs.groupby(["task", "split", "model"], sort=False):
        ttype = g["task_type"].iloc[0]
        for metric in _metrics_for(ttype):
            x = g[metric].values.astype(float)
            lo, hi = t_ci(x)
            blo, bhi = bootstrap_ci(x)
            rows.append({"task": task, "task_type": ttype, "split": split, "model": model, "metric": metric,
                         "n": int(np.sum(~np.isnan(x))), "mean": float(np.nanmean(x)),
                         "std": float(np.nanstd(x, ddof=1)) if len(x) > 1 else np.nan,
                         "ci95_low": lo, "ci95_high": hi, "boot_ci95_low": blo, "boot_ci95_high": bhi,
                         "median": float(np.nanmedian(x)), "min": float(np.nanmin(x)), "max": float(np.nanmax(x))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Paired comparisons
# --------------------------------------------------------------------------------------
def _paired_arrays(a: pd.DataFrame, b: pd.DataFrame, metric: str):
    m = a[PAIR_KEYS + [metric]].merge(b[PAIR_KEYS + [metric]], on=PAIR_KEYS, suffixes=("_a", "_b"))
    m = m.dropna()
    return m[f"{metric}_a"].values.astype(float), m[f"{metric}_b"].values.astype(float)


def paired_tests(x: np.ndarray, y: np.ndarray) -> dict:
    """Paired t-test and Wilcoxon signed-rank test of x versus y (same folds)."""
    d = x - y
    n = len(d)
    out = {"n_pairs": n, "mean_diff": float(d.mean()) if n else np.nan}
    if n >= 2:
        out["diff_ci95_low"], out["diff_ci95_high"] = t_ci(d)
        if np.allclose(d, 0):
            out.update(t_stat=0.0, t_p=1.0, wilcoxon_stat=0.0, wilcoxon_p=1.0)
        else:
            t = ss.ttest_rel(x, y)
            out["t_stat"], out["t_p"] = float(t.statistic), float(t.pvalue)
            try:
                w = ss.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                out["wilcoxon_stat"], out["wilcoxon_p"] = float(w.statistic), float(w.pvalue)
            except ValueError:
                out["wilcoxon_stat"], out["wilcoxon_p"] = np.nan, np.nan
    else:
        out.update(diff_ci95_low=np.nan, diff_ci95_high=np.nan, t_stat=np.nan, t_p=np.nan,
                   wilcoxon_stat=np.nan, wilcoxon_p=np.nan)
    return out


def compare_to_reference(runs: pd.DataFrame, reference: str = "mat", models: list[str] | None = None,
                         alpha: float = 0.05) -> pd.DataFrame:
    """For every (task, split, metric) compare each model against `reference` on paired folds.

    `mean_diff` is model minus reference. `reference_better` states whether the reference wins on the
    metric's natural direction. Holm adjustment is applied within each (task, split, metric) family.
    """
    rows = []
    for (task, split), g in runs.groupby(["task", "split"], sort=False):
        ttype = g["task_type"].iloc[0]
        ref = g[g["model"] == reference]
        if ref.empty:
            continue
        for metric in _metrics_for(ttype):
            fam = []
            for model in (models or [m for m in g["model"].unique() if m != reference]):
                sub = g[g["model"] == model]
                if sub.empty:
                    continue
                x, y = _paired_arrays(sub, ref, metric)
                res = paired_tests(x, y)
                hib = C.HIGHER_IS_BETTER[metric]
                ref_better = (res["mean_diff"] < 0) if hib else (res["mean_diff"] > 0)
                fam.append({"task": task, "task_type": ttype, "split": split, "metric": metric, "model": model,
                            "reference": reference, "model_mean": float(x.mean()) if len(x) else np.nan,
                            "reference_mean": float(y.mean()) if len(y) else np.nan, **res,
                            "reference_better": bool(ref_better)})
            if fam:
                df = pd.DataFrame(fam)
                df["t_p_holm"] = holm(df["t_p"].values)
                df["wilcoxon_p_holm"] = holm(df["wilcoxon_p"].values)
                df["significant_t"] = df["t_p"] < alpha
                df["significant_wilcoxon"] = df["wilcoxon_p"] < alpha
                df["significant_t_holm"] = df["t_p_holm"] < alpha
                df["significant_wilcoxon_holm"] = df["wilcoxon_p_holm"] < alpha
                rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def ablation_contributions(runs: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Contribution of each MAT component = MAT minus the ablation that removes it (paired)."""
    abl = ["mat_nograph", "mat_nodistance", "mat_noattention"]
    comp = compare_to_reference(runs, reference="mat", models=abl, alpha=alpha)
    if comp.empty:
        return comp
    comp = comp.rename(columns={"model": "ablation"})
    comp["component"] = comp["ablation"].map({"mat_nograph": "graph structure", "mat_nodistance": "3D distances",
                                              "mat_noattention": "self-attention"})
    comp["contribution"] = -comp["mean_diff"]                     # MAT minus ablation
    comp["contribution_ci95_low"] = -comp["diff_ci95_high"]
    comp["contribution_ci95_high"] = -comp["diff_ci95_low"]
    return comp


# --------------------------------------------------------------------------------------
# Generalization gap (random minus scaffold)
# --------------------------------------------------------------------------------------
def generalization_gap(runs: pd.DataFrame) -> pd.DataFrame:
    """Random-split score minus Scaffold-split score per (task, model, metric), Welch CI and test.

    Random and scaffold folds are different partitions, so the two samples are independent (unpaired).
    """
    rows = []
    for (task, model), g in runs.groupby(["task", "model"], sort=False):
        ttype = g["task_type"].iloc[0]
        r = g[g["split"] == "random"]
        s = g[g["split"] == "scaffold"]
        if r.empty or s.empty:
            continue
        for metric in _metrics_for(ttype):
            x = r[metric].dropna().values.astype(float)
            y = s[metric].dropna().values.astype(float)
            gap = x.mean() - y.mean()
            se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
            dof = se ** 4 / ((x.var(ddof=1) / len(x)) ** 2 / (len(x) - 1) + (y.var(ddof=1) / len(y)) ** 2 / (len(y) - 1))
            h = ss.t.ppf(0.975, dof) * se if se > 0 else 0.0
            w = ss.ttest_ind(x, y, equal_var=False)
            hib = C.HIGHER_IS_BETTER[metric]
            rows.append({"task": task, "task_type": ttype, "model": model, "metric": metric,
                         "random_mean": float(x.mean()), "scaffold_mean": float(y.mean()), "gap": float(gap),
                         "gap_ci95_low": float(gap - h), "gap_ci95_high": float(gap + h),
                         "welch_t": float(w.statistic), "welch_p": float(w.pvalue), "n_random": len(x), "n_scaffold": len(y),
                         "scaffold_worse": bool(gap > 0) if hib else bool(gap < 0)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Average ranks (as in the MAT paper): rank models per task by the primary metric mean
# --------------------------------------------------------------------------------------
def average_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, g in summary.groupby("split"):
        prim = g[g.apply(lambda r: r["metric"] == C.PRIMARY_METRIC[r["task_type"]], axis=1)]
        piv = prim.pivot(index="task", columns="model", values="mean")
        ranks = pd.DataFrame(index=piv.index)
        for task, row in piv.iterrows():
            hib = C.HIGHER_IS_BETTER[C.PRIMARY_METRIC[C.TASKS[task]["type"]]]
            ranks.loc[task, piv.columns] = row.rank(ascending=not hib)
        for model in piv.columns:
            rows.append({"split": split, "model": model, "avg_rank": float(ranks[model].astype(float).mean()),
                         "n_tasks": int(ranks[model].notna().sum())})
    return pd.DataFrame(rows).sort_values(["split", "avg_rank"]).reset_index(drop=True)


def load_runs(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "error" in df:
        bad = df[~(df["error"].isna() | (df["error"].astype(str) == ""))]
        if len(bad):
            print(f"[stats] WARNING: {len(bad)} failed runs excluded from {path}")
        df = df[df["error"].isna() | (df["error"].astype(str) == "")]
    return df.reset_index(drop=True)
