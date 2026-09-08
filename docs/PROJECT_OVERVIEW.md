# Project overview: what this project is, why it is needed, and what we did

DSC4804 Statistical Learning, Group 14: Yash Yadav (team lead, MAT and attention mechanisms), Kartavya Dev (data engineering and featurisation), Akshat Nagori (statistical learning and baselines).

---

## 1. What the project is

Drug discovery starts by predicting, from a molecule's structure alone, properties such as whether it crosses the blood-brain barrier (BBBP), how soluble it is (ESOL), how strongly it binds to estrogen receptors (Estrogen-α/β), how stable it is against metabolism (MetStab), or its hydration free energy (FreeSolv). These are ADMET-type properties. Good predictions rule out hopeless candidates before any laboratory work.

Several model families compete for this job:

- **Classical QSAR**: fingerprints (ECFP4) fed to Random Forest or Support Vector Machines. Simple, strong, cheap.
- **Graph neural networks (GCN)**: message passing over the molecular graph.
- **Molecule Attention Transformer (MAT)**: a Transformer whose self-attention is mixed, with fixed weights λ, with two chemistry-aware matrices: the normalised adjacency matrix (graph structure) and a kernel of the inter-atomic 3D distances (geometry). MAT's authors also released a checkpoint pretrained on 2 million molecules.

This project asks, with rigorous statistics rather than single lucky runs: **does adding graph structure and 3D geometry to attention actually help, does it help when the test molecules look genuinely different, and how much of MAT's performance is due to pretraining rather than architecture?**

---

## 2. Why it is needed

The literature leaves four gaps that the charter targets:

1. **Pretraining vs architecture** are usually reported together, so nobody knows which matters.
2. **Random splits overstate generalization**: molecules with the same scaffold end up in both training and test. A **Bemis-Murcko scaffold split** puts whole scaffolds out of the training set and gives an honest out-of-distribution estimate.
3. **The value of 3D geometry** is rarely quantified; the only clean way is an ablation that removes exactly one term.
4. **Small-data hybrids** (fingerprints + learned representations) are almost never tested, although small labs live in the small-data regime.

Answering these questions requires a controlled benchmark in which every model sees exactly the same folds, hyperparameters are tuned the same way for everyone, and differences are tested with paired statistics. That is what we built.

---

## 3. What we did, phase by phase

The plan (`PRD.md`) has four phases with three to four parts each. All parts were executed by code in this repository.

### Phase 1: Foundation and data
- **Authentic data.** The seven benchmark CSVs are verbatim copies from the official MAT repository (the DeepChem mirror is offline). ESOL and FreeSolv are z-scored in that release; we recovered the original units exactly by matching every molecule to DeepChem's Delaney file and the Mobley lab's FreeSolv database.
- **Cleaning.** Largest-fragment (salt) stripping, canonical SMILES, duplicate removal with conflicting labels dropped, a 100-heavy-atom cap. Every count is logged (`data/processed/cleaning_log.json`).
- **Featurisation.** The authors' 28-dimensional atom features, adjacency matrices with self-loops, and Euclidean distance matrices from RDKit ETKDGv3 conformers relaxed with UFF (8407 of 8421 molecules embedded in 3D; the authors' 2D fallback covered the remaining 14). ECFP4 fingerprints (2048 bits) for the classical models and the hybrid.
- **Splits.** For each of the five charter seeds, a stratified random 5-fold partition and a balanced scaffold 5-fold partition; within each training portion a seeded 10 % validation hold-out. A defect found during an adversarial code review (ring-free molecules, which have an empty scaffold, were being lumped into one giant fold) was fixed before any experiment ran; the split checker now asserts fold balance and scaffold disjointness.

### Phase 2: Models
- **RF and SVM/SVR** on ECFP4, the SVM with a precomputed Tanimoto kernel.
- **GCN** with PyTorch Geometric, mean+max readout.
- **MAT** re-implemented from the paper and verified bit-for-bit against the authors' code: the fidelity test loads identical weights into both implementations and asserts identical outputs for every kernel and λ setting, and the released 42M-parameter checkpoint loads with every encoder tensor covered.
- **Ablations** by setting one λ to zero and re-normalising the other two: MAT-NoGraph, MAT-NoDistance, MAT-NoAttention. Unit tests prove each ablation is invariant to the removed input (or, for NoAttention, that the query/key projections receive no gradient).
- **ECFP+MAT hybrid**: pooled MAT embedding concatenated with a projected fingerprint, single linear head (same head depth as MAT, so the comparison isolates the fingerprint).
- **Pretrained loader** for the authors' checkpoint.

### Phase 3: Experiments
- One training engine for all deep models (Adam, gradient clipping, early stopping on the validation primary metric, best checkpoint restored) and one scoring path for all models.
- **Hyperparameter policy**: grid search on the seed-42 / fold-0 validation set per (family, task, split), then frozen. Ablations and the hybrid inherit MAT's configuration.
- **Tier 1**: 8 models × 7 tasks × 2 split types × 5 seeds × 5 folds = 2800 runs, executed by 14 parallel worker processes with resume-safe bookkeeping (configuration-hashed run keys).
- **Tier 2**: pretrained vs scratch at the released architecture (42 million parameters). On the CPU-only execution machine one epoch takes about four minutes, so the local run covers seed 42 (both splits, all tasks, equal epoch budget for both variants); the full five-seed protocol is packaged as a Colab GPU notebook.

### Phase 4: Statistics, analysis, reporting
- Mean ± 95 % CI for every metric, model, task and split; paired t-tests and Wilcoxon signed-rank tests of every model against MAT on the 25 paired folds; Holm correction; a seed-level robustness check; ablation contributions; the random-minus-scaffold generalization gap with Welch CIs; MAT-paper-style average ranks; original-unit RMSE/MAE for the regression tasks.
- Attention analysis of trained MAT models: how much learned attention goes to bonded neighbours, how strongly it tracks 3D proximity, its entropy, which atom classes receive attention, and heat-maps for example molecules.
- A generated report (`reports/REPORT.md`) in which every number comes from the results files, and an independent verifier (`scripts/verify_report.py`) that recomputes the key quantities from the raw per-run rows and checks the charter's success criteria (`reports/VERIFICATION.md`).

---

## 4. How correctness was protected

- 40 automated tests (fidelity to the authors' code, ablation invariances, split guarantees, featurisation invariants, end-to-end training, statistics).
- An adversarial multi-agent code review before the expensive runs, which found the scaffold-fold defect and several robustness issues (resume keys ignoring configuration, non-deterministic tuning tie-breaks, divergence handling, thread oversubscription, a hybrid head confound). All confirmed items were fixed and re-tested.
- Everything is seeded (splits, validation hold-outs, conformers, model initialisation, batch order), package versions are pinned, and `python run_all.py` re-creates every artefact.

---

## 5. Results and findings

The numbers live in `reports/REPORT.md` (generated) and are summarised, with interpretation, in Section 6 below once the pipeline has finished. The report is organised as: data (1), protocol (2), main results per split (3), statistical comparison against MAT (4), ablations (5), generalization gap (6), pretrained vs scratch (7), hybrid (8), attention (9), verdict on the hypothesis (10), limitations (11), reproducibility (12).

---

## 6. Interpretation

_This section is completed after the benchmark finishes; see the end of this file._

---

## 7. Limitations, stated plainly

- CPU-only execution forced small MAT configurations (64–128 dimensions, 2–4 layers) and a reduced local protocol for the 42M-parameter comparison. The GPU notebook removes the second limitation.
- For ESOL and FreeSolv one scaffold group (benzene) is larger than a fold, so one scaffold test fold is the same for every seed; seed-level tests are reported as the conservative check.
- Cross-validation fold scores overlap in training data, so paired p-values are optimistic; Holm-adjusted and seed-level p-values are reported next to the raw ones.
- The scaffold split's validation hold-out is random within the training portion, so early stopping is tuned on in-distribution molecules.

---

## 8. Team responsibilities mapped to the code

| Member | Charter responsibility | Modules |
|---|---|---|
| Yash Yadav | MAT implementation, attention mechanisms, milestones | `models/mat.py`, `models/hybrid.py`, `attention.py`, `tests/test_mat_fidelity.py`, PRD |
| Kartavya Dev | Preprocessing, RDKit conformers, distance matrices, splits | `data.py`, `featurize.py`, `scripts/01_prepare_data.py`, `tests/test_data_and_metrics.py` |
| Akshat Nagori | RF/SVM/GCN, CV, hypothesis tests, ablations | `models/classical.py`, `models/gcn.py`, `stats.py`, `scripts/05_stats.py`, `tests/test_stats.py` |

AI-tool usage (charter section F): Claude Code was used to draft code and documents under the team's direction; every artefact is reviewed by the team and all results are regenerated by the scripts in this repository.
