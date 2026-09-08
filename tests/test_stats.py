"""Tests for the statistical evaluation module."""
import numpy as np
import pandas as pd
from scipy import stats as ss

from molbench import stats as S


def _fake_runs(seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for task, ttype in [("bbbp", "clf"), ("esol", "reg")]:
        for split in ["random", "scaffold"]:
            for model, shift in [("mat", 0.0), ("rf", -0.05), ("mat_nograph", -0.02), ("mat_nodistance", 0.0),
                                 ("mat_noattention", -0.03)]:
                for s in [42, 123, 456, 789, 1011]:
                    for f in range(5):
                        base = 0.85 if split == "random" else 0.75
                        row = {"task": task, "task_type": ttype, "split": split, "model": model, "seed": s, "fold": f}
                        if ttype == "clf":
                            row.update(roc_auc=base + shift + rng.randn() * 0.01, pr_auc=base + shift + rng.randn() * 0.01,
                                       f1=base + shift + rng.randn() * 0.01, rmse=np.nan, mae=np.nan, r2=np.nan)
                        else:
                            row.update(rmse=0.6 - shift + rng.randn() * 0.02, mae=0.4 - shift + rng.randn() * 0.02,
                                       r2=0.8 + shift + rng.randn() * 0.02, roc_auc=np.nan, pr_auc=np.nan, f1=np.nan)
                        rows.append(row)
    return pd.DataFrame(rows)


def test_t_ci_matches_scipy():
    x = np.array([0.8, 0.82, 0.79, 0.85, 0.81])
    lo, hi = S.t_ci(x)
    ref = ss.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=ss.sem(x))
    assert np.isclose(lo, ref[0]) and np.isclose(hi, ref[1])


def test_holm():
    p = np.array([0.01, 0.04, 0.03, np.nan])
    adj = S.holm(p)
    assert np.isclose(adj[0], 0.03) and np.isclose(adj[2], 0.06) and np.isclose(adj[1], 0.06) and np.isnan(adj[3])
    assert np.all(adj[:3] >= p[:3])


def test_summary_and_comparisons():
    runs = _fake_runs()
    summ = S.summarize(runs)
    assert set(summ["n"]) == {25}
    assert ((summ["ci95_low"] <= summ["mean"]) & (summ["mean"] <= summ["ci95_high"])).all()
    comp = S.compare_to_reference(runs, reference="mat")
    assert set(comp["n_pairs"]) == {25}
    assert ((comp["t_p"] >= 0) & (comp["t_p"] <= 1)).all()
    rf_bbbp = comp[(comp["model"] == "rf") & (comp["task"] == "bbbp") & (comp["metric"] == "roc_auc")]
    assert (rf_bbbp["t_p"] < 0.05).all() and rf_bbbp["reference_better"].all()
    assert (rf_bbbp["mean_diff"] < 0).all()
    # regression: rf has higher rmse (worse) -> mean_diff > 0 and reference better
    rf_esol = comp[(comp["model"] == "rf") & (comp["task"] == "esol") & (comp["metric"] == "rmse")]
    assert (rf_esol["mean_diff"] > 0).all() and rf_esol["reference_better"].all()
    abl = S.ablation_contributions(runs)
    g = abl[(abl["ablation"] == "mat_nograph") & (abl["task"] == "bbbp") & (abl["metric"] == "roc_auc")]
    assert (g["contribution"] > 0).all()
    gap = S.generalization_gap(runs)
    m = gap[(gap["model"] == "mat") & (gap["task"] == "bbbp") & (gap["metric"] == "roc_auc")]
    assert np.isclose(m["gap"].iloc[0], 0.10, atol=0.02) and m["scaffold_worse"].all()
    ranks = S.average_ranks(summ)
    best = ranks[ranks["split"] == "random"].iloc[0]["model"]
    assert best in ("mat", "mat_nodistance")


def test_paired_tests_identical_inputs():
    x = np.ones(10)
    r = S.paired_tests(x, x)
    assert r["t_p"] == 1.0 and r["wilcoxon_p"] == 1.0
