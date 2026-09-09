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

Every number below is taken from `reports/REPORT.md`, which is generated from `results/` and checked by `scripts/verify_report.py` (542 checks, 0 failures). Execution record: Tier 1 = 2800 runs (2087 on Kaggle T4 GPUs, 713 on the laptop CPU), Tier 2 = 140 runs on Kaggle T4 GPUs, 0 failed runs, about 49 hours of model fitting summed over runs.

**Main results (primary metric, mean over 25 folds; ROC-AUC for classification, z-scored RMSE for regression).**

| Task (split) | RF | SVM/SVR | GCN | MAT | MAT-NoGraph | MAT-NoDistance | MAT-NoAttention | ECFP+MAT |
|---|---|---|---|---|---|---|---|---|
| BBBP (random) | 0.921 | 0.921 | 0.899 | 0.892 | 0.883 | 0.887 | 0.855 | 0.911 |
| BBBP (scaffold) | 0.902 | 0.904 | 0.859 | 0.876 | 0.875 | 0.873 | 0.850 | 0.888 |
| ESOL (random) ↓ | 0.574 | 0.485 | 0.457 | 0.385 | 0.411 | 0.391 | 0.453 | 0.491 |
| ESOL (scaffold) ↓ | 0.773 | 0.631 | 0.510 | 0.499 | 0.524 | 0.493 | 0.510 | 0.590 |
| FreeSolv (random) ↓ | 0.588 | 0.471 | 0.338 | 0.369 | 0.376 | 0.380 | 0.458 | 0.379 |
| FreeSolv (scaffold) ↓ | 0.648 | 0.577 | 0.426 | 0.431 | 0.446 | 0.424 | 0.477 | 0.437 |
| Estrogen-α (random) | 0.968 | 0.969 | 0.956 | 0.953 | 0.947 | 0.952 | 0.925 | 0.969 |
| Estrogen-α (scaffold) | 0.952 | 0.954 | 0.949 | 0.937 | 0.929 | 0.932 | 0.912 | 0.948 |
| Estrogen-β (random) | 0.926 | 0.929 | 0.913 | 0.893 | 0.878 | 0.888 | 0.852 | 0.922 |
| Estrogen-β (scaffold) | 0.898 | 0.897 | 0.883 | 0.852 | 0.845 | 0.858 | 0.833 | 0.889 |
| MetStab-high (random) | 0.902 | 0.904 | 0.848 | 0.791 | 0.778 | 0.789 | 0.734 | 0.898 |
| MetStab-high (scaffold) | 0.870 | 0.861 | 0.774 | 0.751 | 0.736 | 0.752 | 0.696 | 0.851 |
| MetStab-low (random) | 0.873 | 0.874 | 0.847 | 0.765 | 0.755 | 0.771 | 0.645 | 0.863 |
| MetStab-low (scaffold) | 0.810 | 0.818 | 0.784 | 0.743 | 0.726 | 0.745 | 0.672 | 0.805 |

In original units, MAT's ESOL RMSE is 0.81 log10(mol/L) on the random split (RF 1.20) and 1.05 on the scaffold split (RF 1.62); its FreeSolv RMSE is 1.42 kcal/mol (RF 2.26) and 1.66 kcal/mol (RF 2.49).

**Statistical comparisons against MAT (Wilcoxon signed-rank on 25 paired folds, p < 0.05).**

- RF and SVM/SVR beat the small scratch MAT on all five classification tasks under both splits; MAT beats them on both regression tasks under both splits. Count: MAT significantly better on 2/7 tasks, the classical models on 5/7, for each split.
- GCN beats MAT on the MetStab tasks and Estrogen-β (and FreeSolv, random split); MAT beats GCN on ESOL (random) and BBBP (scaffold).
- The ECFP+MAT hybrid beats MAT on every classification task (5/7 random, 4/7 scaffold significant; BBBP scaffold p = 0.055) and closes most of the gap to RF/SVM (for example MetStab-high random 0.898 vs RF 0.902); MAT beats the hybrid on ESOL under both splits. FreeSolv shows no significant difference.
- Ablations: removing **self-attention** hurts significantly on 7/7 tasks (random) and 5/7 (scaffold); removing **graph structure** hurts significantly on 5/7 (random) and 3/7 (scaffold) with effects of about 0.01 ROC-AUC; removing **3D distances** changes nothing significantly on any task or split (every |Δ| ≤ 0.011). The seed-level robustness tests agree in direction.
- Generalization gap (random minus scaffold ROC-AUC, mean over the five classification tasks): MAT 0.027, MAT-NoDistance 0.025, MAT-NoGraph 0.026, RF 0.032, SVM 0.033, hybrid 0.036, GCN 0.043, MAT-NoAttention 0.009 (but it is also the weakest model). On the regression tasks the scaffold split increases every model's RMSE, least for MAT on ESOL (+0.114 vs +0.198 for RF).
- Pretrained vs scratch at the released 42M-parameter architecture (5 seeds, equal 15-epoch budget): pretraining improves the point estimate on 7/7 tasks (random) and 6/7 (scaffold), significantly by the paired t-test on Estrogen-α (both splits) and both MetStab endpoints (random); the largest gains are on MetStab (+0.05 to +0.07 ROC-AUC). ESOL under the scaffold split is the exception (pretrained worse, high variance). With this short fine-tuning budget the large models do not beat the small scratch MAT trained to convergence on most tasks.
- Attention analysis (MAT on BBBP and ESOL): the learned self-attention is **not** local. Its share on bonded neighbours (0.07 to 0.14) is below the uniform baseline (0.09 to 0.17), its Spearman correlation with inverse distance is about −0.03 to −0.07, and it concentrates on the dummy node (up to 0.33 of the mass on ESOL) and on heteroatoms (2 to 4 times the attention received by aromatic carbons in layer 1). Locality comes entirely from the injected adjacency and distance terms: the fused attention has a bonded share of 0.33 to 0.35 and a distance correlation of 0.6 to 0.7.

---

## 6. Interpretation

**Does graph structure and 3D geometry help attention?** Partly. The learned self-attention is the single most valuable term (removing it costs 0.02 to 0.12 ROC-AUC and 0.05 to 0.09 RMSE in z-units). Graph structure adds a small but consistent gain on classification (about 0.01 ROC-AUC, significant on five of seven tasks under the random split). The 3D distance term adds nothing measurable at this scale: the adjacency term already supplies locality, and the attention analysis shows that the learned heads use the freedom the distance term is supposed to give them for global, heteroatom-centred patterns rather than proximity. The charter's hypothesis is therefore supported for graph structure and for attention itself, and not supported for 3D distances.

**Does the structure improve out-of-distribution generalization?** Modestly. MAT and its structured ablations lose less ROC-AUC from random to scaffold split (about 0.026) than RF, SVM (0.032 to 0.033) or GCN (0.043), and on ESOL MAT degrades far less than RF. But a smaller gap does not make MAT the best out-of-distribution model: on every classification task the fingerprint models still score higher in absolute terms under the scaffold split.

**Which model should a practitioner use?** For the physical-chemistry regressions (solubility, hydration free energy), the geometry-aware transformer is clearly best: a third lower RMSE than RF in original units. For the bioactivity and ADME classifications, a well-tuned Random Forest or Tanimoto SVM on ECFP4 remains the strongest and cheapest choice, and a small MAT trained from scratch does not match it. The late-fusion hybrid is the practical compromise: it recovers almost all of the fingerprint models' classification accuracy while keeping most of MAT's regression accuracy, which is exactly the low-data recipe the charter hoped for.

**What does pretraining buy?** Relative to the same architecture trained from scratch with the same short budget, pretraining helps on almost every task and most on the small, imbalanced MetStab endpoints. In absolute terms, at a 15-epoch budget the 42M-parameter models do not beat a 100k-parameter MAT trained to convergence, so the reported strength of pretrained MAT in the literature depends on longer fine-tuning than a course-project budget allows. Separating these two statements is precisely the point of the pretrained-versus-scratch comparison.

**Caveats that matter for reading the numbers.** The MAT models in Tier 1 are small (64 to 128 dimensions, 2 to 4 layers) because of the CPU-first budget, so the comparison is "small MAT from scratch" versus "tuned classical models"; the fingerprint models' advantage on classification may shrink with the paper-scale architecture. The scaffold folds are seed-dependent except for the benzene scaffold group on ESOL and FreeSolv, which is larger than a fold; seed-level tests are reported for that reason. Cross-validation folds share training data, so paired p-values are optimistic and Holm-adjusted values are reported alongside.

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
