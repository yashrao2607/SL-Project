"""Phase 4.3: regenerate every figure from the results CSVs."""
from __future__ import annotations

import json

import pandas as pd

from molbench import config as C
from molbench import plotting as P


def main():
    C.ensure_dirs()
    F = C.RESULTS_FIGURES
    made = []
    s1 = C.RESULTS_SUMMARY / "summary_tier1.csv"
    if s1.exists():
        summ = pd.read_csv(s1)
        for split in C.SPLITS:
            made.append(P.plot_performance(summ, split, F / f"fig_performance_{split}.png"))
        abl = pd.read_csv(C.RESULTS_STATS / "ablations_tier1.csv")
        if not abl.empty:
            made.append(P.plot_ablations(abl, F / "fig_ablations.png"))
        gap = pd.read_csv(C.RESULTS_STATS / "generalization_gap_tier1.csv")
        if not gap.empty:
            made.append(P.plot_generalization_gap(gap, F / "fig_generalization_gap.png"))
        ranks = pd.read_csv(C.RESULTS_STATS / "average_ranks_tier1.csv")
        if not ranks.empty:
            made.append(P.plot_average_ranks(ranks, F / "fig_average_ranks.png"))
    s2 = C.RESULTS_SUMMARY / "summary_tier2.csv"
    if s2.exists():
        summ2 = pd.read_csv(s2)
        comp = pd.read_csv(C.RESULTS_STATS / "pretrained_vs_scratch.csv")
        made.append(P.plot_pretrained_vs_scratch(summ2, comp, F / "fig_pretrained_vs_scratch.png"))
    for hs in sorted(C.RESULTS_ATTENTION.glob("*_head_stats.csv")):
        task, split = hs.name.replace("_head_stats.csv", "").rsplit("_", 1)
        stats = pd.read_csv(hs)
        base = json.loads((C.RESULTS_ATTENTION / f"{task}_{split}_uniform_baseline.json").read_text())
        made.append(P.plot_attention_heads(stats, base, task, split, F / f"fig_attention_heads_{task}_{split}.png"))
        rec = pd.read_csv(C.RESULTS_ATTENTION / f"{task}_{split}_received_by_class.csv")
        made.append(P.plot_received_by_class(rec, task, split, F / f"fig_attention_classes_{task}_{split}.png"))
        npz = C.RESULTS_ATTENTION / f"{task}_{split}_examples.npz"
        if npz.exists():
            made.append(P.plot_attention_examples(npz, task, split, F / f"fig_attention_examples_{task}_{split}.png"))
    for m in made:
        print("[figures]", m)


if __name__ == "__main__":
    main()
