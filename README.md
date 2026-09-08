# Statistical Evaluation and Extension of the Molecule Attention Transformer (MAT)

DSC4804 Statistical Learning, School of Engineering & Technology. Group 14: Yash Yadav (team lead, MAT and attention), Kartavya Dev (data engineering and featurisation), Akshat Nagori (baselines and statistics).

This repository is a complete, reproducible benchmark of the **Molecule Attention Transformer** (Maziarka et al., 2020) against classical QSAR models, a graph neural network, three MAT ablations, an ECFP+MAT late-fusion hybrid and the authors' pretrained MAT, on seven molecular property tasks, under both random and Bemis-Murcko scaffold cross-validation, with paired statistical testing. Everything, from raw CSV to the final report, is produced by scripts; no number is typed by hand.

Documents to read, in order:

| Document | What it is |
|---|---|
| `SL_Group_14_Project_Charter.pdf` | The signed project charter this work implements |
| `PRD.md` | Product requirements document: scope, data, protocol, four phases with parts and acceptance checks |
| `docs/PROJECT_OVERVIEW.md` | Plain-language overview: what the project is, why it is needed, what was done, what was found |
| `docs/CODE_GUIDE.md` | Every file and folder: purpose, dependencies, use case |
| `reports/REPORT.md` | The results report (generated from `results/`) |
| `reports/VERIFICATION.md` | Machine check that the report numbers match the raw runs and the charter criteria |

---

## 1. What is benchmarked

| Family | Model | Input | Implementation |
|---|---|---|---|
| Classical QSAR | Random Forest | ECFP4 (2048-bit Morgan, radius 2) | scikit-learn |
| Classical QSAR | SVM / SVR | ECFP4 with a precomputed Tanimoto kernel | scikit-learn |
| Graph neural network | GCN (Kipf & Welling) | atom features + bonds | PyTorch Geometric |
| Transformer | MAT (trained from scratch) | atom features + adjacency + 3D distance matrix | faithful re-implementation, verified against the authors' code |
| Ablations | MAT-NoGraph, MAT-NoDistance, MAT-NoAttention | same as MAT with one attention term removed | λ triplets (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5) |
| Hybrid | ECFP + MAT late fusion | MAT embedding ⊕ projected ECFP4, linear head | `models/hybrid.py` |
| Pretraining | MAT-large pretrained vs MAT-large scratch | authors' released 42M-parameter checkpoint vs same architecture randomly initialised | `models/mat.py::load_pretrained` |

Tasks (all from the official MAT repository, `data/raw/`): BBBP, ESOL, FreeSolv, Estrogen-α, Estrogen-β, MetStab-high, MetStab-low (five classification, two regression).

Protocol: seeds 42, 123, 456, 789, 1011 × 5-fold CV × {random, scaffold} splits = 50 partitions per task, 72 % train / 8 % validation / 20 % test, identical for all models. Metrics: ROC-AUC, PR-AUC, F1 (classification); RMSE, MAE, R² (regression). Statistics: 95 % CIs, paired t-tests, Wilcoxon signed-rank tests, Holm correction, seed-level robustness check, generalization gap. Full details: `PRD.md` section 3.

---

## 2. Installation

Tested on Windows 11, Python 3.11.9, CPU only (16 cores, 16 GB RAM). Linux/macOS work the same way; a CUDA GPU is used automatically when present.

```bash
git clone <your repository url>
cd "SL Project"
python -m venv .venv && .venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                                       # installs the molbench package (needed by the worker processes)
python -m pytest tests -q                              # 40 tests, ~30 s
```

`requirements.txt` pins the exact versions used for the reported results. The editable install is required: the benchmark runs in spawned worker processes that must be able to `import molbench`.

Two files are not in the repository and are re-created by the scripts:

- `data/pretrained/mat_pretrained_weights.pt` (168 MB, authors' checkpoint, Google Drive id `11-TZj8tlnD7ykQGliO9bCrySJNBnYD2k`), download with
  `python -m gdown "https://drive.google.com/uc?id=11-TZj8tlnD7ykQGliO9bCrySJNBnYD2k" -O data/pretrained/mat_pretrained_weights.pt`
- `data/processed/features/mol_cache.pkl` (97 MB, ETKDG conformers and matrices), created by `scripts/01_prepare_data.py` in about six minutes with 12 processes.

---

## 3. Running

One command reproduces everything (the order is tests → data → tuning → Tier 1 → Tier 2 → statistics → attention → figures → report → verification):

```bash
python run_all.py                 # full protocol; several hours on a 16-core CPU (see timings below)
python run_all.py --smoke         # end-to-end check on FreeSolv with tiny budgets (~10 minutes)
python run_all.py --skip-tier2    # everything except the 42M-parameter pretrained-vs-scratch runs
```

Every step is resume-safe: completed runs are stored in `results/raw/*.csv` keyed by (model, task, split, seed, fold, configuration hash) and are skipped on re-run, so an interrupted run continues where it stopped.

Individual steps:

| Step | Command | Output | Approximate time (this machine) |
|---|---|---|---|
| Phase 1 | `python scripts/01_prepare_data.py --jobs 12` | `data/processed/*.csv`, `splits/*.json`, `features/mol_cache.pkl`, `cleaning_log.json` | 6 min |
| 3.2 tuning | `python scripts/02_tune.py` | `results/tuning/tuning_runs.csv`, `best_configs.json/csv` | 25 min |
| 3.3 Tier 1 | `python scripts/03_run_tier1.py` | `results/raw/tier1_runs.csv` (2800 rows) | 4–7 h (14 workers) |
| 3.4 Tier 2 | `python scripts/04_run_tier2.py --local` | `results/raw/tier2_runs.csv` | ~4 h (CPU, seed 42) or ~1 h for all 5 seeds on a Colab GPU |
| 4.1 statistics | `python scripts/05_stats.py` | `results/summary/*.csv`, `results/stats/*.csv` | seconds |
| 4.2 attention | `python scripts/06_attention.py` | `results/attention/*`, `results/checkpoints/*.pt` | 10 min |
| 4.3 figures | `python scripts/07_figures.py` | `results/figures/*.png` | seconds |
| 4.4 report | `python scripts/08_report.py` then `python scripts/verify_report.py` | `reports/REPORT.md`, `report_numbers.json`, `VERIFICATION.md` | seconds |

Useful options: `--workers N --threads T` on the tuning/Tier-1 scripts (default 14 × 1), `--models`, `--tasks` to run subsets, `--max-epochs`, `--patience`.

### Tier 2 on a Colab GPU (recommended for the full 5-seed protocol)

The 42M-parameter model needs about four minutes per epoch on this CPU, so the local run covers seed 42 only. For the full protocol:

1. `python scripts/make_colab_bundle.py` creates `molbench_colab_bundle.zip` (code + processed data + splits + features).
2. Open `notebooks/colab_tier2.ipynb` in Colab with a GPU runtime, upload the zip when prompted, run all cells.
3. Download the produced `results/raw/tier2_runs.csv` into the local `results/raw/` and re-run steps 4.1, 4.3 and 4.4.

The notebook downloads the pretrained checkpoint itself and runs the fidelity tests before training. No credentials are required.

---

## 4. Repository layout

```
SL Project/
├── SL_Group_14_Project_Charter.pdf   signed charter
├── PRD.md                            product requirements document (phases, parts, acceptance checks)
├── README.md                         this file
├── docs/                             PROJECT_OVERVIEW.md, CODE_GUIDE.md
├── requirements.txt, pyproject.toml  pinned environment, package definition
├── run_all.py                        one-command reproduction
├── src/molbench/                     the Python package
│   ├── config.py                     paths, seeds, task and model registries
│   ├── data.py                       loading, cleaning, scaffolds, split generation
│   ├── featurize.py                  ECFP4, atom features, ETKDG conformers, adjacency/distance matrices
│   ├── metrics.py                    ROC-AUC, PR-AUC, F1, RMSE, MAE, R²
│   ├── models/                       classical.py, gcn.py, mat.py, hybrid.py
│   ├── grids.py                      hyperparameter grids
│   ├── engine.py                     training / evaluation loop
│   ├── runner.py                     parallel, resume-safe grid execution
│   ├── stats.py                      CIs, paired tests, Holm, ablations, generalization gap, ranks
│   ├── attention.py                  attention-map statistics
│   └── plotting.py                   figures
├── scripts/                          numbered pipeline steps 01–08, verify_report.py, make_colab_bundle.py
├── tests/                            pytest suite (fidelity, models, data, engine, statistics)
├── notebooks/colab_tier2.ipynb       GPU notebook for the pretrained-vs-scratch runs
├── reference/mat_original/           authors' MAT source (fidelity test only)
├── data/                             raw (verbatim), processed (cleaned, splits, features), pretrained
├── results/                          tuning, raw, summary, stats, attention, figures, checkpoints
└── reports/                          REPORT.md, report_numbers.json, VERIFICATION.md
```

---

## 5. Data provenance and cleaning

All seven CSVs are verbatim copies from `https://github.com/ardigen/MAT/tree/master/data` (the benchmark of the MAT paper). ESOL and FreeSolv labels in that release are z-scored; the original units are recovered exactly by matching molecules to DeepChem's `delaney-processed.csv` and the Mobley lab's FreeSolv `database.txt` (both stored in `data/raw/_original_units/`), giving y = 2.0955·z − 3.0501 log10(mol/L) for ESOL and y = 3.8448·z − 3.8030 kcal/mol for FreeSolv.

Cleaning (identical for every task, logged in `data/processed/cleaning_log.json`): parse with RDKit, keep the largest fragment, canonicalise, remove duplicates (conflicting labels dropped), remove molecules above 100 heavy atoms. 3D geometry: RDKit ETKDGv3 + UFF, heavy atoms only, with the authors' 2D fallback (used for 14 of 8421 molecules).

---

## 6. Correctness safeguards

- `tests/test_mat_fidelity.py`: the re-implemented MAT gives the same output as the authors' `transformer.py` for identical weights and inputs (all kernels and λ settings), and the pretrained checkpoint loads with every encoder tensor covered.
- `tests/test_models.py`: ablations are provably invariant to the removed input (NoGraph ignores adjacency, NoDistance ignores distances, NoAttention gives zero gradient to Q/K), padding does not change outputs, attention rows sum to one.
- `tests/test_data_and_metrics.py`: cleaning policy, scaffold-disjoint and balanced folds, featurisation invariants, Tanimoto kernel, metric values.
- `tests/test_engine.py`: every model family trains end-to-end on both task types; regression metrics are in label units; same seed gives identical results.
- `scripts/verify_report.py`: recomputes report numbers from the raw run CSVs with independent code and checks the charter's success criteria.

---

## 7. Charter success criteria → where they are met

| Criterion | Where |
|---|---|
| ROC-AUC, PR-AUC, F1 for classification; RMSE, MAE, R² for regression | `results/summary/summary_tier1.csv`, report section 3 |
| Generalization gap random vs scaffold | `results/stats/generalization_gap_tier1.csv`, report section 6 |
| Rigorous statistical testing, 5-fold CV × seeds 42, 123, 456, 789, 1011 | `results/raw/tier1_runs.csv` (2800 rows) |
| Paired t-tests and Wilcoxon signed-rank tests, p < 0.05 | `results/stats/vs_mat_tier1.csv`, `ablations_tier1.csv`, report sections 4–5 |
| 95 % confidence intervals | every summary and comparison CSV |
| Ablations NoGraph / NoDistance / NoAttention | report section 5 |
| Pretrained vs scratch MAT | report section 7 |
| ECFP+MAT hybrid | report section 8 |
| Attention analysis | report section 9, `results/attention/` |

---

## 8. Acknowledgements

MAT architecture and datasets: Maziarka, Ł., Danel, T., Mucha, S., Rataj, K., Tabor, J., Jastrzębski, S. (2020). *Molecule Attention Transformer*. arXiv:2002.08264. Code in `reference/mat_original/` is the authors' MIT-licensed implementation, kept only for the fidelity test. FreeSolv: Mobley & Guthrie (2014). ESOL: Delaney (2004). MoleculeNet: Wu et al. (2018).
