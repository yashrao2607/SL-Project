"""Figures (Phase 4.3). Static matplotlib PNGs regenerated from the results CSVs.

Colour policy: one fixed colour per model (identity never cycles), MAT family in blue, sequential
single-hue ramp for attention heat-maps, recessive grid, thin marks, 95% CI error bars.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from rdkit import Chem  # noqa: E402

from . import config as C  # noqa: E402

MODEL_COLORS = {
    "mat": "#2a78d6", "rf": "#eb6834", "svm": "#1baf7a", "gcn": "#eda100", "hybrid": "#e87ba4",
    "mat_nograph": "#008300", "mat_nodistance": "#4a3aa7", "mat_noattention": "#e34948",
    "mat_large_pretrained": "#0d366b", "mat_large_scratch": "#86b6ef",
}
COMPONENT_COLORS = {"graph structure": "#008300", "3D distances": "#4a3aa7", "self-attention": "#e34948"}
SPLIT_COLORS = {"random": "#2a78d6", "scaffold": "#eb6834"}
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"
TASK_LABELS = {"bbbp": "BBBP", "esol": "ESOL", "freesolv": "FreeSolv", "estrogen-alpha": "Estrogen-α",
               "estrogen-beta": "Estrogen-β", "metstab-high": "MetStab-high", "metstab-low": "MetStab-low"}
METRIC_LABELS = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1", "rmse": "RMSE", "mae": "MAE", "r2": "R²"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 150,
})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _primary_rows(summary: pd.DataFrame):
    return summary[summary.apply(lambda r: r["metric"] == C.PRIMARY_METRIC[r["task_type"]], axis=1)]


# --------------------------------------------------------------------------------------
def plot_performance(summary: pd.DataFrame, split: str, out: Path, models=C.TIER1_MODELS):
    """One panel per task: primary metric mean with 95% CI per model (horizontal bars)."""
    prim = _primary_rows(summary[summary["split"] == split])
    tasks = [t for t in C.TASK_NAMES if t in set(prim["task"])]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2))
    axes = axes.ravel()
    for ax, task in zip(axes, tasks):
        sub = prim[prim["task"] == task].set_index("model")
        ms = [m for m in models if m in sub.index]
        y = np.arange(len(ms))
        means = sub.loc[ms, "mean"].values
        lo = sub.loc[ms, "ci95_low"].values
        hi = sub.loc[ms, "ci95_high"].values
        ax.barh(y, means, color=[MODEL_COLORS[m] for m in ms], height=0.62, edgecolor="white", linewidth=1)
        ax.errorbar(means, y, xerr=[means - lo, hi - means], fmt="none", ecolor=TEXT, elinewidth=1, capsize=2)
        for yi, v in zip(y, means):
            ax.text(v, yi, f" {v:.3f}", va="center", ha="left", fontsize=7.5, color=TEXT)
        ax.set_yticks(y)
        ax.set_yticklabels([C.MODEL_LABELS[m] for m in ms], fontsize=7.5)
        ax.invert_yaxis()
        metric = C.PRIMARY_METRIC[C.TASKS[task]["type"]]
        ax.set_title(f"{TASK_LABELS[task]}  ({METRIC_LABELS[metric]}{' ↓' if not C.HIGHER_IS_BETTER[metric] else ' ↑'})")
        ax.grid(axis="y", visible=False)
        if metric == "roc_auc":
            ax.set_xlim(max(0.4, (lo.min() - 0.05)), min(1.0, hi.max() + 0.06))
        else:
            ax.set_xlim(0, hi.max() * 1.18)
    for ax in axes[len(tasks):]:
        ax.axis("off")
    fig.suptitle(f"{split.capitalize()} split: primary metric, mean and 95% CI over 5 seeds × 5 folds", fontsize=11,
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out)


def plot_ablations(abl: pd.DataFrame, out: Path):
    """Forest plot: contribution of each MAT term (MAT minus ablation) per task, both splits."""
    comps = ["graph structure", "3D distances", "self-attention"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, split in zip(axes, C.SPLITS):
        sub = abl[(abl["split"] == split)]
        sub = sub[sub.apply(lambda r: r["metric"] == C.PRIMARY_METRIC[r["task_type"]], axis=1)]
        tasks = [t for t in C.TASK_NAMES if t in set(sub["task"])]
        for ti, task in enumerate(tasks):
            for ci, comp in enumerate(comps):
                r = sub[(sub["task"] == task) & (sub["component"] == comp)]
                if r.empty:
                    continue
                r = r.iloc[0]
                sign = 1 if C.HIGHER_IS_BETTER[r["metric"]] else -1     # positive = component helps
                y = ti + (ci - 1) * 0.25
                c, lo, hi = sign * r["contribution"], sign * r["contribution_ci95_low"], sign * r["contribution_ci95_high"]
                lo, hi = min(lo, hi), max(lo, hi)
                ax.plot([lo, hi], [y, y], color=COMPONENT_COLORS[comp], linewidth=2, solid_capstyle="round")
                mk = "o" if r["significant_wilcoxon"] else "o"
                ax.scatter([c], [y], color=COMPONENT_COLORS[comp], s=32 if r["significant_wilcoxon"] else 22,
                           marker=mk, edgecolor="white" if r["significant_wilcoxon"] else COMPONENT_COLORS[comp],
                           facecolor=COMPONENT_COLORS[comp] if r["significant_wilcoxon"] else "white", zorder=3)
        ax.axvline(0, color=TEXT2, linewidth=0.8)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([TASK_LABELS[t] for t in tasks])
        ax.invert_yaxis()
        ax.set_title(f"{split.capitalize()} split")
        ax.set_xlabel("Contribution to primary metric (MAT − ablation; + = term helps; ROC-AUC units or −RMSE)")
        ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=COMPONENT_COLORS[c], marker="o", linewidth=2, label=c) for c in comps]
    handles.append(plt.Line2D([], [], color=TEXT2, marker="o", markerfacecolor="white", linewidth=0, label="hollow = not significant (Wilcoxon p ≥ 0.05)"))
    axes[0].legend(handles=handles, loc="lower left", fontsize=8)
    fig.suptitle("Ablation study: what each MAT term contributes (95% CI over 25 paired folds)", fontsize=11,
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out)


def plot_generalization_gap(gap: pd.DataFrame, out: Path, models=C.TIER1_MODELS):
    """Dot + CI per model and task: random-split score minus scaffold-split score (primary metric)."""
    sub = gap[gap.apply(lambda r: r["metric"] == C.PRIMARY_METRIC[r["task_type"]], axis=1)]
    tasks = [t for t in C.TASK_NAMES if t in set(sub["task"])]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2))
    axes = axes.ravel()
    for ax, task in zip(axes, tasks):
        s = sub[sub["task"] == task].set_index("model")
        ms = [m for m in models if m in s.index]
        metric = C.PRIMARY_METRIC[C.TASKS[task]["type"]]
        sign = 1 if C.HIGHER_IS_BETTER[metric] else -1        # positive = worse under scaffold
        for i, m in enumerate(ms):
            g, lo, hi = sign * s.loc[m, "gap"], sign * s.loc[m, "gap_ci95_low"], sign * s.loc[m, "gap_ci95_high"]
            lo, hi = min(lo, hi), max(lo, hi)
            ax.plot([lo, hi], [i, i], color=MODEL_COLORS[m], linewidth=2, solid_capstyle="round")
            ax.scatter([g], [i], color=MODEL_COLORS[m], s=28, zorder=3)
            ax.text(hi, i, f"  {g:+.3f}", va="center", fontsize=7, color=TEXT)
        ax.axvline(0, color=TEXT2, linewidth=0.8)
        ax.set_yticks(range(len(ms)))
        ax.set_yticklabels([C.MODEL_LABELS[m] for m in ms], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_title(f"{TASK_LABELS[task]} ({METRIC_LABELS[metric]})")
        ax.grid(axis="y", visible=False)
    for ax in axes[len(tasks):]:
        ax.axis("off")
    fig.suptitle("Generalization gap = random-split score − scaffold-split score (positive = worse out-of-distribution), 95% CI",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out)


def plot_pretrained_vs_scratch(summary2: pd.DataFrame, comp: pd.DataFrame, out: Path):
    prim = _primary_rows(summary2)
    tasks = [t for t in C.TASK_NAMES if t in set(prim["task"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
    for ax, split in zip(axes, C.SPLITS):
        x = np.arange(len(tasks))
        for j, m in enumerate(["mat_large_scratch", "mat_large_pretrained"]):
            s = prim[(prim["split"] == split) & (prim["model"] == m)].set_index("task")
            means = np.array([s.loc[t, "mean"] if t in s.index else np.nan for t in tasks])
            lo = np.array([s.loc[t, "ci95_low"] if t in s.index else np.nan for t in tasks])
            hi = np.array([s.loc[t, "ci95_high"] if t in s.index else np.nan for t in tasks])
            ax.bar(x + (j - 0.5) * 0.36, means, width=0.34, color=MODEL_COLORS[m], label=C.MODEL_LABELS[m],
                   edgecolor="white", linewidth=1)
            ax.errorbar(x + (j - 0.5) * 0.36, means, yerr=[means - lo, hi - means], fmt="none", ecolor=TEXT,
                        elinewidth=1, capsize=2)
            for xi, v in zip(x + (j - 0.5) * 0.36, means):
                if not np.isnan(v):
                    ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=TEXT)
        for i, t in enumerate(tasks):
            r = comp[(comp["split"] == split) & (comp["task"] == t)]
            r = r[r.apply(lambda q: q["metric"] == C.PRIMARY_METRIC[q["task_type"]], axis=1)]
            if not r.empty:
                p = r.iloc[0]["wilcoxon_p"]
                ax.text(i, ax.get_ylim()[1] * 0.98, f"p={p:.3f}" if not np.isnan(p) else "", ha="center", va="top",
                        fontsize=7, color=TEXT2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{TASK_LABELS[t]}\n({METRIC_LABELS[C.PRIMARY_METRIC[C.TASKS[t]['type']]]})" for t in tasks], fontsize=7.5)
        ax.set_title(f"{split.capitalize()} split")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Pretrained MAT vs scratch MAT at the released architecture (42M parameters), 5 seeds, fold 0; Wilcoxon p",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out)


def plot_average_ranks(ranks: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, split in zip(axes, C.SPLITS):
        s = ranks[ranks["split"] == split].sort_values("avg_rank")
        y = np.arange(len(s))
        ax.barh(y, s["avg_rank"], color=[MODEL_COLORS[m] for m in s["model"]], height=0.62, edgecolor="white")
        for yi, v in zip(y, s["avg_rank"]):
            ax.text(v, yi, f" {v:.2f}", va="center", fontsize=8, color=TEXT)
        ax.set_yticks(y)
        ax.set_yticklabels([C.MODEL_LABELS[m] for m in s["model"]], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{split.capitalize()} split")
        ax.set_xlabel("Average rank across 7 tasks (1 = best)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Average rank of each model (primary metric)", fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out)


# --------------------------------------------------------------------------------------
# Attention figures
# --------------------------------------------------------------------------------------
def plot_attention_heads(head_stats: pd.DataFrame, baseline: dict, task: str, split: str, out: Path):
    s = head_stats[head_stats["kind"] == "self_attn"]
    layers = sorted(s["layer"].unique())
    heads = sorted(s["head"].unique())
    stats = [("bonded_share", "Attention on bonded neighbours"), ("dist_corr_spearman", "Spearman(attention, 1/distance)"),
             ("entropy", "Row entropy (nats)"), ("dummy_share", "Attention on dummy node")]
    fig, axes = plt.subplots(1, len(stats), figsize=(4 * len(stats), 3.6))
    ramp = ["#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
    for ax, (col, title) in zip(axes, stats):
        x = np.arange(len(heads))
        for li, layer in enumerate(layers):
            v = s[s["layer"] == layer].sort_values("head")[col].values
            ax.bar(x + (li - (len(layers) - 1) / 2) * 0.8 / len(layers), v, width=0.8 / len(layers),
                   color=ramp[li % len(ramp)], label=f"layer {layer + 1}", edgecolor="white", linewidth=0.8)
        if col in baseline:
            ax.axhline(baseline[col], color="#e34948", linewidth=1.2, linestyle="--")
            ax.text(len(heads) - 0.5, baseline[col], " uniform", color="#e34948", fontsize=7, va="bottom", ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels([f"head {h + 1}" for h in heads], fontsize=8)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Learned self-attention statistics, MAT on {TASK_LABELS[task]} ({split} split, test set)",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out)


def plot_attention_examples(npz_path: Path, task: str, split: str, out: Path, layer: int = -1, n_examples: int = 3):
    d = np.load(npz_path, allow_pickle=True)
    smiles = d["smiles"][:n_examples]
    fig, axes = plt.subplots(n_examples, 3, figsize=(12, 3.7 * n_examples))
    if n_examples == 1:
        axes = axes[None, :]
    for r, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        labels = ["·"] + [f"{a.GetSymbol()}{a.GetIdx() + 1}" for a in mol.GetAtoms()]
        sa = d["self_attn"][r][layer]          # (h, n, n)
        fused = d["attn"][r][layer]
        adj = d["adj"][r]
        n = sa.shape[-1]
        panels = [(sa.mean(0), "self-attention (mean of heads)"), (fused.mean(0), "fused attention (mean of heads)"),
                  (adj / np.maximum(adj.sum(-1, keepdims=True), 1e-6), "row-normalised adjacency")]
        for c, (M, title) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(M, cmap=SEQ_BLUE, vmin=0, vmax=max(0.05, float(np.percentile(M, 99))))
            bi, bj = np.nonzero(np.triu(adj, 1))
            if c < 2:
                ax.scatter(bj, bi, s=6, color="#e34948", marker="s", linewidth=0)
                ax.scatter(bi, bj, s=6, color="#e34948", marker="s", linewidth=0)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, fontsize=6, rotation=90)
            ax.set_yticklabels(labels, fontsize=6)
            ax.grid(False)
            ax.set_title(f"{title}", fontsize=8.5)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        axes[r, 0].set_ylabel(f"{smi}\npred={float(d['pred'][r]):.2f} true={float(d['true'][r]):.2f}", fontsize=7)
    fig.suptitle(f"Attention maps (last layer) for test molecules, MAT on {TASK_LABELS[task]} ({split}); red squares = bonds",
                 fontsize=11, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out)


def plot_received_by_class(rec: pd.DataFrame, task: str, split: str, out: Path):
    layers = sorted(rec["layer"].unique())
    classes = [c for c in rec["atom_class"].unique()]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(classes))
    ramp = ["#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
    for li, layer in enumerate(layers):
        v = [rec[(rec["layer"] == layer) & (rec["atom_class"] == c)]["mean_attention_received"].values[0] for c in classes]
        ax.bar(x + (li - (len(layers) - 1) / 2) * 0.8 / len(layers), v, width=0.8 / len(layers),
               color=ramp[li % len(ramp)], label=f"layer {layer + 1}", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_ylabel("Mean attention received per atom")
    ax.legend(fontsize=7)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Attention received by atom class, MAT on {TASK_LABELS[task]} ({split})")
    fig.tight_layout()
    return _save(fig, out)


def load_json(p: Path):
    return json.loads(Path(p).read_text())
