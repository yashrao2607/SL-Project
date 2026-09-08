# Product Requirements Document (PRD)

## Statistical Evaluation and Extension of the Molecule Attention Transformer (MAT)

| Field | Value |
|---|---|
| Course | DSC4804 Statistical Learning, School of Engineering & Technology |
| Group | 14 (Yash Yadav, Kartavya Dev, Akshat Nagori) |
| Source charter | `SL_Group_14_Project_Charter.pdf` (dated 23 Aug 2026) |
| PRD version | 1.0, 9 Sep 2026 |
| Execution platform | Windows 11, Python 3.11, PyTorch 2.7 (CPU only), 16 cores, 16 GB RAM |

---

## 1. Purpose and scope

The charter asks for a rigorous, statistically validated benchmark of the Molecule Attention Transformer (MAT) against classical and graph baselines, a set of MAT ablations, a pretrained-versus-scratch comparison, an ECFP+MAT late-fusion hybrid, and an attention analysis, evaluated under both Random (i.i.d.) and Bemis-Murcko Scaffold (out-of-distribution) splits on the MAT benchmark datasets.

This PRD converts the charter into an executable plan with four phases. Every phase has three or four parts, each with a concrete deliverable and an acceptance check. All code lives in one reproducible Python package so the whole study can be re-run with a single command.

### 1.1 Central hypothesis (from the charter)

Augmenting self-attention with molecular graph structure and 3D inter-atomic distances genuinely improves prediction accuracy and out-of-distribution generalization, rather than adding complexity without benefit.

### 1.2 In scope

- Seven prediction tasks from the six charter datasets (MetStab has two endpoints, high and low, exactly as released by the MAT authors).
- Models: RF, SVM/SVR, GCN, MAT (scratch), MAT-NoGraph, MAT-NoDistance, MAT-NoAttention, ECFP+MAT hybrid, pretrained MAT vs scratch MAT at the pretrained architecture.
- Splits: Random 5-fold CV and Scaffold 5-fold CV, each repeated over seeds 42, 123, 456, 789, 1011.
- Metrics: ROC-AUC, PR-AUC, F1 (classification); RMSE, MAE, R² (regression).
- Statistics: paired t-tests, Wilcoxon signed-rank tests, 95% confidence intervals, generalization gap, significance at p < 0.05.
- Attention-weight analysis of trained MAT models.
- Report with tables and figures, plus a machine-checked verification of every reported number.

### 1.3 Out of scope

- Re-running MAT self-supervised pretraining (we use the authors' released checkpoint).
- GPU-specific optimisation (no CUDA device available; everything runs on CPU).
- SMILES-based transformers (discussed in the literature review only; not benchmarked, per the charter's model list).

---

## 2. Data sources (authentic, verified)

All datasets are taken verbatim from the official MAT repository (`ardigen/MAT`, `data/` folder), which is exactly the benchmark used in the MAT paper (Maziarka et al., 2020, arXiv:2002.08264). Files were downloaded on 9 Sep 2026 into `data/raw/`.

| Task | File | Molecules | Task type | Label | Notes |
|---|---|---|---|---|---|
| BBBP | `bbbp/bbbp.csv` | 2039 | Classification | blood-brain barrier penetration | MoleculeNet; 64 canonical duplicates, 105 multi-fragment SMILES |
| ESOL | `esol/esol.csv` | 1128 | Regression | aqueous solubility (z-scored by MAT authors) | MoleculeNet (Delaney); 11 duplicates |
| FreeSolv | `freesolv/freesolv.csv` | 642 | Regression | hydration free energy (z-scored by MAT authors) | MoleculeNet (SAMPL) |
| Estrogen-α | `estrogen-alpha/estrogen-alpha.csv` | 2398 | Classification | ER-α activity | MAT benchmark; one 457-heavy-atom outlier |
| Estrogen-β | `estrogen-beta/estrogen-beta.csv` | 1961 | Classification | ER-β activity | MAT benchmark |
| MetStab-high | `mesta-high/mesta-high.csv` | 2127 | Classification | high metabolic stability | MAT benchmark (MetStabOn); 12.3 % positives |
| MetStab-low | `mesta-low/mesta-low.csv` | 2127 | Classification | low metabolic stability | same molecules as MetStab-high |

Pretrained MAT weights: the authors' released checkpoint (`pretrained_weights.pt`, 168 MB, Google Drive id `11-TZj8tlnD7ykQGliO9bCrySJNBnYD2k`), stored at `data/pretrained/mat_pretrained_weights.pt`. Verified architecture: 8 layers, d_model 1024, 16 heads, d_atom 28, 42,077,212 parameters.

Data-cleaning policy (applied identically to every dataset, fully logged):

1. Parse SMILES with RDKit; drop unparsable molecules (0 found).
2. Keep the largest organic fragment (salt / counter-ion stripping).
3. Canonicalise SMILES; drop exact duplicates. Conflicting labels for the same canonical SMILES are dropped entirely.
4. Drop molecules with more than 100 heavy atoms (affects only a handful of extreme outliers; count is logged).
5. Regression labels: metrics are computed on the released z-scored labels. Where the original-unit scale can be recovered from the MoleculeNet source (by exact molecule matching), RMSE/MAE are additionally reported in original units.

---

## 3. Experimental protocol

### 3.1 Splits

- Seeds: 42, 123, 456, 789, 1011. Each seed produces its own 5-fold partition.
- Random split: stratified K-fold (classification) or plain K-fold (regression) with the seed as `random_state`.
- Scaffold split: Bemis-Murcko scaffolds (RDKit `MurckoScaffold`, chirality ignored). Scaffold groups are shuffled with the seed and assigned greedily to the currently smallest fold, so test folds contain scaffolds never seen during training. This is a "balanced scaffold" K-fold and yields a genuine out-of-distribution test.
- Inside each training portion, 10 % is held out (seeded) as a validation set for early stopping and model selection. Final proportions: 72 % train / 8 % validation / 20 % test.
- Every model sees exactly the same folds, so all comparisons are paired.

### 3.2 Hyperparameter policy (fairness)

Each model family is tuned on the validation set of seed 42, fold 0, separately for every (task, split type). The winning configuration is then frozen and used for all 25 runs of that (task, split). This gives every family the same tuning budget style and prevents test-set leakage.

| Family | Grid |
|---|---|
| RF | n_estimators {500}, max_features {sqrt, 0.3}, min_samples_leaf {1, 3}, class_weight {None, balanced} (clf only) |
| SVM / SVR | Tanimoto kernel on ECFP4; C {0.1, 1, 10}; class_weight {None, balanced} (clf only); SVR epsilon {0.05, 0.1, 0.2} |
| GCN | hidden {64, 128}; layers {2, 3}; lr 1e-3; dropout 0.1; batch 64 |
| MAT (scratch) | (d_model, N) ∈ {(64, 2), (64, 4), (128, 2)}; h 4; lr {5e-4, 1e-3}; dropout 0.1; batch 64; λ = (0.33, 0.33, 0.34). The (128, 4) cell was dropped after timing (3× the cost of (64, 2)) to keep the 2800-run CPU protocol tractable |
| MAT ablations | inherit MAT's frozen configuration, λ of the removed term set to 0 and the remaining two renormalised to 0.5 each |
| ECFP+MAT hybrid | inherit MAT's configuration; ECFP projected to 128-d (ReLU), concatenated with the pooled MAT embedding, single linear output layer (same head depth as MAT) |
| Pretrained MAT vs scratch MAT | fixed pretrained architecture (1024 / 8 / 16); lr 1e-4 for both; batch 32; identical epoch budget (15 epochs, patience 5 on GPU; 8 epochs, patience 3 for the local CPU run) |

### 3.3 Compute tiers (CPU-only machine)

| Tier | Models | Protocol | Runs |
|---|---|---|---|
| 1 (full) | RF, SVM/SVR, GCN, MAT, 3 ablations, hybrid | 5 seeds × 5 folds × 2 splits × 7 tasks | 8 × 350 = 2800 |
| 2 (reduced, 42M-parameter model) | pretrained MAT, scratch MAT at pretrained size | 5 seeds × fold 0 × 2 splits × 7 tasks on a GPU (`notebooks/colab_tier2.ipynb`); locally (CPU, ~4 min per epoch) seed 42 × fold 0 × 2 splits × 7 tasks | 140 (GPU) / 28 (local CPU) |

Measured on the execution machine: one epoch of the 42M-parameter model takes about 4 minutes with 8 threads, so the local Tier 2 is limited to seed 42; the notebook reproduces the 5-seed protocol on a free Colab GPU in about an hour. Nothing in Tier 1 is reduced. Deep models train for at most 60 epochs with early-stopping patience 10 (validation primary metric); parallel execution uses 14 single-thread workers.

### 3.4 Metrics and statistics

- Classification: ROC-AUC, PR-AUC (average precision), F1 at threshold 0.5.
- Regression: RMSE, MAE, R².
- Per (task, split, model): mean, standard deviation and 95 % CI (t-distribution over the 25 fold scores; bootstrap CI as a check).
- Pairwise tests against MAT (and MAT against each ablation): paired t-test and Wilcoxon signed-rank test over the 25 paired fold scores. Significance at p < 0.05; Holm-adjusted p-values reported alongside raw ones.
- Generalization gap: Random-split score minus Scaffold-split score per model and task, with its own CI.

### 3.5 Attention analysis

Trained scratch MAT models (seed 42, fold 0) on BBBP and ESOL are used to extract per-head attention. Reported: share of attention on bonded neighbours versus non-bonded atoms, correlation of attention with inverse 3D distance per head, attention entropy per head, attention received by atom classes (aromatic, heteroatoms, ring atoms, dummy node), and heat-maps for representative molecules.

---

## 4. Phases and parts

### Phase 1: Foundation and data

| Part | Deliverable | Acceptance check |
|---|---|---|
| 1.1 Environment and repository scaffold | `requirements.txt`, package `src/molbench/`, `scripts/`, `results/`, `reports/`, `README.md` | `python -c "import molbench"` succeeds; all versions pinned |
| 1.2 Data acquisition and cleaning | `data/raw/*` (verbatim), `data/processed/<task>.csv`, `data/processed/cleaning_log.json` | every SMILES parses; no duplicate canonical SMILES; per-task counts logged |
| 1.3 Featurisation | ECFP4 (2048 bits), atom-feature matrices, adjacency matrices, ETKDG conformers with UFF relaxation, Euclidean distance matrices, cached to `data/processed/features/` | shapes consistent; distance matrices symmetric with zero diagonal; conformer failure rate logged |
| 1.4 Split generation | `data/processed/splits/<task>_<split>_seed<s>.json` with 5 folds each | folds disjoint and exhaustive; scaffold test folds share zero scaffolds with their training folds |

### Phase 2: Models

| Part | Deliverable | Acceptance check |
|---|---|---|
| 2.1 Classical baselines | `models/classical.py`: RF and Tanimoto-kernel SVM/SVR with grid search | fits and predicts on every task; kernel is symmetric with unit diagonal |
| 2.2 GCN baseline | `models/gcn.py` (PyTorch Geometric GCNConv, mean+max readout) | unit test: forward pass shapes and gradient flow |
| 2.3 MAT reproduction | `models/mat.py`: faithful PyTorch re-implementation | fidelity test: identical outputs to the original `transformer.py` given identical weights and inputs; pretrained checkpoint loads with zero missing encoder keys |
| 2.4 Ablations, hybrid, pretrained loader | λ-controlled ablations, `models/hybrid.py`, `load_pretrained()` | unit tests for each variant; ablation λ's sum to 1 |

### Phase 3: Experiments

| Part | Deliverable | Acceptance check |
|---|---|---|
| 3.1 Training and evaluation engine | `engine.py`: seeded training loop, early stopping, metric computation, per-run CSV rows | one full end-to-end run per model on FreeSolv completes and writes a row |
| 3.2 Hyperparameter selection | `results/tuning/*.csv` with the chosen configuration per (family, task, split) | every family has a frozen configuration before Tier 1 starts |
| 3.3 Tier 1 benchmark | `results/raw/tier1_runs.csv` (2800 rows) | row count equals plan; no NaN metrics; each run logged with seed, fold, split, model, wall-time |
| 3.4 Tier 2 pretrained vs scratch | `results/raw/tier2_runs.csv` | encoder weights verified loaded; paired rows for pretrained and scratch on identical folds |

### Phase 4: Statistics, analysis, and reporting

| Part | Deliverable | Acceptance check |
|---|---|---|
| 4.1 Statistical evaluation | `results/summary/*.csv`, `results/stats/*.csv` (t-tests, Wilcoxon, CIs, Holm, generalization gap) | every test uses exactly 25 paired scores; p-values in [0, 1]; CI contains the mean |
| 4.2 Attention analysis | `results/attention/*.csv`, heat-map figures | per-head statistics reproducible from saved attention tensors |
| 4.3 Figures and tables | `results/figures/*.png` (performance bars with CIs, ablation deltas, generalization gap, attention) | every figure regenerates from CSVs by one script |
| 4.4 Final report, documentation and verification | `reports/REPORT.md`, `reports/VERIFICATION.md`, `README.md` (detailed usage), `docs/CODE_GUIDE.md` (every file and folder: purpose, dependencies, use case), `docs/PROJECT_OVERVIEW.md` (what, why, what was done), Colab notebook for Tier 2 | every number in the report is re-derived from CSVs by `scripts/verify_report.py`; charter success-criteria checklist all ticked |

---

## 5. Repository layout

```
SL Project/
├── SL_Group_14_Project_Charter.pdf
├── PRD.md                      (this document)
├── README.md
├── requirements.txt
├── run_all.py                  (one-command reproduction)
├── data/
│   ├── raw/                    (verbatim MAT-repo CSVs)
│   ├── processed/              (cleaned CSVs, features, splits)
│   └── pretrained/             (authors' MAT checkpoint)
├── reference/mat_original/     (authors' transformer.py, data_utils.py; used only for the fidelity test)
├── src/molbench/
│   ├── data.py                 (loading, cleaning, scaffolds, splits)
│   ├── featurize.py            (ECFP, atom features, ETKDG conformers, matrices)
│   ├── metrics.py
│   ├── models/{classical,gcn,mat,hybrid}.py
│   ├── engine.py               (training / evaluation)
│   ├── stats.py                (tests, CIs, gap)
│   ├── attention.py
│   └── plotting.py
├── scripts/                    (numbered pipeline steps)
├── tests/                      (pytest)
├── results/{tuning,raw,summary,stats,attention,figures}/
└── reports/{REPORT.md,VERIFICATION.md}
```

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| CPU-only training of the 42M-parameter pretrained MAT is slow | Tier 2 reduced protocol; batch size and epoch budget measured before launch; results clearly labelled |
| Conformer generation fails for some molecules | Fallback to 2D coordinates exactly as in the original MAT code; failures logged and counted |
| Scaffold K-fold produces unbalanced folds | Greedy smallest-fold assignment; fold sizes logged |
| Multiple comparisons inflate false positives | Holm correction reported next to raw p-values |
| Reproducibility drift | All seeds fixed; features cached; versions pinned; `run_all.py` re-creates everything |

---

## 7. Definition of done

1. All 2800 Tier 1 runs and all Tier 2 runs present in `results/raw/` with no NaN metrics.
2. Summary tables with mean ± 95 % CI for every metric, model, task and split.
3. Paired t-test and Wilcoxon results for every model versus MAT and every ablation versus MAT, under both splits.
4. Generalization-gap table and figure.
5. Attention analysis tables and figures.
6. `reports/REPORT.md` written from those tables, with `scripts/verify_report.py` confirming each number.
6a. `README.md`, `docs/CODE_GUIDE.md` and `docs/PROJECT_OVERVIEW.md` written and consistent with the code and results.
7. Fresh-clone reproduction check: `python run_all.py --smoke` completes end-to-end.
