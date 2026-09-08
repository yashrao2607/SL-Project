"""Data loading, cleaning, Bemis-Murcko scaffolds and split generation."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from . import config as C

RDLogger.DisableLog("rdApp.*")


# --------------------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------------------
def load_raw(task: str) -> pd.DataFrame:
    """Read the verbatim CSV from the MAT repository (two columns: smiles, y)."""
    path = C.DATA_RAW / C.TASKS[task]["file"]
    df = pd.read_csv(path)
    df = df.iloc[:, :2].copy()
    df.columns = ["smiles", "y"]
    df["smiles"] = df["smiles"].astype(str).str.strip()
    return df


def largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) == 1:
        return mol
    return max(frags, key=lambda m: (m.GetNumHeavyAtoms(), Chem.MolToSmiles(m)))


def clean_dataset(df: pd.DataFrame, task_type: str, max_heavy_atoms: int = C.MAX_HEAVY_ATOMS):
    """Apply the cleaning policy of PRD section 2 and return (clean_df, log).

    Steps: parse -> largest fragment -> canonicalise -> drop >max_heavy_atoms -> de-duplicate
    (identical labels keep one row; conflicting labels are dropped entirely).
    """
    log = {"n_raw": int(len(df)), "n_unparsable": 0, "n_multi_fragment": 0, "n_too_large": 0,
           "n_duplicate_rows_removed": 0, "n_conflicting_label_molecules_dropped": 0}
    rows = []
    for smi, y in zip(df["smiles"], df["y"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            log["n_unparsable"] += 1
            continue
        if "." in smi:
            log["n_multi_fragment"] += 1
            mol = largest_fragment(mol)
        can = Chem.MolToSmiles(mol)
        if mol.GetNumHeavyAtoms() > max_heavy_atoms:
            log["n_too_large"] += 1
            continue
        rows.append((can, float(y), smi))
    tmp = pd.DataFrame(rows, columns=["smiles", "y", "original_smiles"])

    groups = tmp.groupby("smiles")
    keep = []
    for can, g in groups:
        ys = g["y"].values
        if task_type == "clf":
            conflict = len(set(ys)) > 1
        else:
            conflict = (ys.max() - ys.min()) > 1e-6
        if conflict:
            log["n_conflicting_label_molecules_dropped"] += 1
            log["n_duplicate_rows_removed"] += int(len(g))
            continue
        log["n_duplicate_rows_removed"] += int(len(g) - 1)
        keep.append(g.iloc[0])
    clean = pd.DataFrame(keep).reset_index(drop=True)
    clean["y"] = clean["y"].astype(float)
    if task_type == "clf":
        assert set(clean["y"].unique()) <= {0.0, 1.0}, "classification labels must be binary"
    clean["scaffold"] = [bemis_murcko_scaffold(s) for s in clean["smiles"]]
    clean["n_heavy"] = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in clean["smiles"]]
    log["n_clean"] = int(len(clean))
    log["n_scaffolds"] = int(clean["scaffold"].nunique())
    if task_type == "clf":
        log["positive_rate"] = float(clean["y"].mean())
    else:
        log["y_mean"] = float(clean["y"].mean())
        log["y_std"] = float(clean["y"].std())
    return clean, log


def processed_path(task: str) -> Path:
    return C.DATA_PROCESSED / f"{task}.csv"


def load_processed(task: str) -> pd.DataFrame:
    df = pd.read_csv(processed_path(task))
    df["scaffold"] = df["scaffold"].fillna("")          # ring-free molecules have an empty scaffold
    return df


# --------------------------------------------------------------------------------------
# Scaffolds
# --------------------------------------------------------------------------------------
def bemis_murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:  # pragma: no cover - extremely rare RDKit failures
        scaf = ""
    return scaf


# --------------------------------------------------------------------------------------
# Split generation
# --------------------------------------------------------------------------------------
def random_folds(y: np.ndarray, task_type: str, seed: int, k: int = C.N_FOLDS) -> list[np.ndarray]:
    idx = np.arange(len(y))
    if task_type == "clf":
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return [test for _, test in splitter.split(idx, y.astype(int))]
    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [test for _, test in splitter.split(idx)]


def scaffold_folds(scaffolds: list[str], seed: int, k: int = C.N_FOLDS) -> list[np.ndarray]:
    """Balanced scaffold K-fold (chemprop-style ordering, extended to K folds).

    * Molecules sharing a Bemis-Murcko scaffold form one indivisible group.
    * Ring-free molecules have no scaffold (empty string); each one is its own group, otherwise
      they would collapse into a single giant block (half of FreeSolv) and wreck the fold balance.
    * Groups larger than n / (2k) ("large") are placed first, in seeded random order, then the
      remaining groups in seeded random order; every group goes to the currently smallest fold.
    Hence each seed yields a different scaffold partition and every test fold is scaffold-disjoint
    from its training folds.
    """
    n = len(scaffolds)
    groups = defaultdict(list)
    for i, s in enumerate(scaffolds):
        key = s if isinstance(s, str) and s else f"acyclic:{i}"
        groups[key].append(i)
    group_list = list(groups.values())
    rng = np.random.RandomState(seed)
    rng.shuffle(group_list)
    threshold = n / (2 * k)
    large = [g for g in group_list if len(g) > threshold]
    small = [g for g in group_list if len(g) <= threshold]
    folds = [[] for _ in range(k)]
    sizes = np.zeros(k, dtype=int)
    for g in large + small:
        j = int(np.argmin(sizes))          # smallest fold; ties resolved by lowest index
        folds[j].extend(g)
        sizes[j] += len(g)
    return [np.array(sorted(f), dtype=int) for f in folds]


def make_splits(df: pd.DataFrame, task_type: str, split: str, seed: int, k: int = C.N_FOLDS,
                val_fraction: float = C.VAL_FRACTION) -> list[dict]:
    """Return a list of k dicts with train / val / test index arrays (as python lists)."""
    y = df["y"].values
    if split == "random":
        tests = random_folds(y, task_type, seed, k)
    elif split == "scaffold":
        tests = scaffold_folds(df["scaffold"].tolist(), seed, k)
    else:
        raise ValueError(split)
    all_idx = np.arange(len(df))
    out = []
    for fold, test in enumerate(tests):
        rest = np.setdiff1d(all_idx, test)
        strat = y[rest].astype(int) if task_type == "clf" else None
        try:
            train, val = train_test_split(rest, test_size=val_fraction, random_state=seed + fold, stratify=strat)
        except ValueError:  # too few members of a class for stratification
            train, val = train_test_split(rest, test_size=val_fraction, random_state=seed + fold)
        out.append({"fold": fold, "train": sorted(map(int, train)), "val": sorted(map(int, val)),
                    "test": sorted(map(int, test))})
    return out


def split_path(task: str, split: str, seed: int) -> Path:
    return C.SPLIT_DIR / f"{task}_{split}_seed{seed}.json"


def save_splits(task: str, split: str, seed: int, folds: list[dict]) -> None:
    split_path(task, split, seed).write_text(json.dumps(folds))


def load_splits(task: str, split: str, seed: int) -> list[dict]:
    return json.loads(split_path(task, split, seed).read_text())


def check_splits(df: pd.DataFrame, folds: list[dict], split: str, balance_tol: float | None = 1.5) -> None:
    """Raise if folds are not disjoint / exhaustive / scaffold-disjoint / badly unbalanced."""
    n = len(df)
    tests = [set(f["test"]) for f in folds]
    union = set().union(*tests)
    assert union == set(range(n)), "test folds must cover every molecule exactly once"
    assert sum(len(t) for t in tests) == n, "test folds overlap"
    scaf = df["scaffold"].fillna("").values
    sizes = [len(t) for t in tests]
    if balance_tol is not None:
        assert max(sizes) <= balance_tol * n / len(folds), f"unbalanced test folds: {sizes}"
        assert min(sizes) >= n / (balance_tol * len(folds)), f"unbalanced test folds: {sizes}"
    for f in folds:
        tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
        assert not (tr & va) and not (tr & te) and not (va & te), "train/val/test overlap"
        assert tr | va | te == set(range(n)), "train+val+test must be exhaustive"
        if split == "scaffold":
            train_scaf = {x for x in scaf[list(tr | va)] if x}      # "" = no ring system = no scaffold
            test_scaf = {x for x in scaf[list(te)] if x}
            assert not (train_scaf & test_scaf), "scaffold leakage between train and test"


# --------------------------------------------------------------------------------------
# Original-unit recovery for the z-scored regression labels
# --------------------------------------------------------------------------------------
def recover_regression_scale(task: str, clean: pd.DataFrame) -> dict | None:
    """Estimate the linear map y_orig = a * y_z + b from matched molecules in the MoleculeNet source.

    Returns None when the source file is not present. `a` equals the standard deviation of the
    original labels (so RMSE_orig = a * RMSE_z), `b` their mean.
    """
    src_dir = C.DATA_RAW / "_original_units"
    if task == "esol":
        f = src_dir / "delaney-processed.csv"
        if not f.exists():
            return None
        src = pd.read_csv(f)
        smiles_col = "smiles"
        y_col = "measured log solubility in mols per litre"
        unit = "log10(mol/L)"
    elif task == "freesolv":
        f = src_dir / "freesolv_database.txt"
        if not f.exists():
            return None
        rows = []
        for line in f.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = [p.strip() for p in line.split(";")]
            rows.append((parts[1], float(parts[3])))
        src = pd.DataFrame(rows, columns=["smiles", "expt"])
        smiles_col, y_col, unit = "smiles", "expt", "kcal/mol"
    else:
        return None
    canon = {}
    for s, v in zip(src[smiles_col], src[y_col]):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        m = largest_fragment(m)
        canon[Chem.MolToSmiles(m)] = float(v)
    matched = clean[clean["smiles"].isin(canon)]
    if len(matched) < 50:
        return None
    yz = matched["y"].values
    yo = np.array([canon[s] for s in matched["smiles"]])
    a, b = np.polyfit(yz, yo, 1)
    resid = yo - (a * yz + b)
    return {"task": task, "unit": unit, "scale_a": float(a), "offset_b": float(b),
            "n_matched": int(len(matched)), "max_abs_residual": float(np.abs(resid).max()),
            "source": f.name}
