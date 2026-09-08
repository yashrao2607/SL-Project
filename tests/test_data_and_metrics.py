"""Tests for cleaning, scaffolds, splits, featurisation invariants, Tanimoto kernel and metrics."""
import numpy as np
import pandas as pd

from molbench import data as D
from molbench.featurize import featurize_smiles
from molbench.metrics import classification_metrics, regression_metrics
from molbench.models.classical import tanimoto_kernel


def test_clean_dataset_policy():
    df = pd.DataFrame({
        "smiles": ["CCO", "OCC", "[Na+].CC(=O)[O-]", "CC(=O)[O-]", "c1ccccc1", "c1ccccc1", "C1CCCCC1", "xyz"],
        "y": [1, 1, 0, 1, 1, 1, 0, 1],
    })
    clean, log = D.clean_dataset(df, "clf")
    assert log["n_unparsable"] == 1
    assert log["n_multi_fragment"] == 1
    # CCO/OCC identical label -> one row; acetate appears twice with conflicting labels -> dropped
    assert "CCO" in set(clean["smiles"])
    assert "CC(=O)[O-]" not in set(clean["smiles"])
    assert log["n_conflicting_label_molecules_dropped"] == 1
    assert len(clean) == 3
    assert "scaffold" in clean and "n_heavy" in clean


def test_scaffold_folds_disjoint_and_exhaustive():
    smiles = ["c1ccccc1" + "C" * i for i in range(10)] + ["C1CCCCC1" + "O" * i for i in range(7)] + \
             ["CC" * (i + 1) for i in range(8)]
    df = pd.DataFrame({"smiles": smiles, "y": np.arange(len(smiles)) % 2})
    clean, _ = D.clean_dataset(df, "clf")
    for split in ["random", "scaffold"]:
        folds = D.make_splits(clean, "clf", split, 42)
        D.check_splits(clean, folds, split, balance_tol=None)   # 25 molecules: balance guard not meaningful


def test_acyclic_molecules_are_spread_across_scaffold_folds():
    smiles = ["C" * (i + 2) for i in range(40)] + ["c1ccccc1" + "C" * i for i in range(10)] + ["c1ccncc1" + "N" * i for i in range(5)]
    df = pd.DataFrame({"smiles": smiles, "y": np.arange(len(smiles)) % 2})
    clean, _ = D.clean_dataset(df, "clf")
    assert (clean["scaffold"] == "").sum() == 40
    folds = D.make_splits(clean, "clf", "scaffold", 42)
    D.check_splits(clean, folds, "scaffold")
    acyclic_per_fold = [int((clean["scaffold"].values[f["test"]] == "").sum()) for f in folds]
    assert min(acyclic_per_fold) >= 4, acyclic_per_fold
    sizes = [len(f["test"]) for f in folds]
    assert max(sizes) - min(sizes) <= 10, sizes


def test_scaffold_split_is_seed_dependent_but_deterministic():
    smiles = ["c1ccccc1" + "C" * i for i in range(10)] + ["C1CCCCC1" + "O" * i for i in range(7)] + \
             ["CC" * (i + 1) for i in range(8)] + ["c1ccncc1" + "N" * i for i in range(5)]
    df = pd.DataFrame({"smiles": smiles, "y": np.arange(len(smiles)) % 2})
    clean, _ = D.clean_dataset(df, "clf")
    a = D.make_splits(clean, "clf", "scaffold", 42)
    b = D.make_splits(clean, "clf", "scaffold", 42)
    c = D.make_splits(clean, "clf", "scaffold", 123)
    assert a == b
    assert a != c


def test_featurisation_invariants():
    r = featurize_smiles("CC(=O)Oc1ccccc1C(=O)O")   # aspirin, 13 heavy atoms
    assert r["afm"].shape == (14, 28) and r["adj"].shape == (14, 14) and r["dist"].shape == (14, 14)
    assert r["afm"][0, 0] == 1 and r["afm"][0, 1:].sum() == 0          # dummy node indicator
    assert np.allclose(r["dist"], r["dist"].T)
    assert np.allclose(np.diag(r["dist"])[1:], 0)
    assert np.all(r["dist"][0, :] == 1e6) and np.all(r["dist"][:, 0] == 1e6)
    assert np.all(np.diag(r["adj"])[1:] == 1) and r["adj"][0].sum() == 0
    assert r["adj"][1:, 1:].sum() == 13 + 2 * 13                       # self loops + 13 bonds x 2
    assert r["ecfp"].shape == (2048,) and r["ecfp"].dtype == np.uint8
    assert r["conformer"] == "etkdg"
    # bonded atoms are within a chemically sensible distance
    bonded = np.argwhere(np.triu(r["adj"][1:, 1:], 1) > 0)
    for i, j in bonded:
        assert 1.0 < r["dist"][1 + i, 1 + j] < 2.0


def test_tanimoto_kernel():
    A = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)
    K = tanimoto_kernel(A, A)
    assert K.shape == (3, 3)
    assert np.allclose(np.diag(K)[:2], 1.0)
    assert np.isclose(K[0, 1], 1 / 3)
    assert K[2, 2] == 0.0 and np.allclose(K, K.T)


def test_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = classification_metrics(y, p)
    assert np.isclose(m["roc_auc"], 0.75)
    assert 0 <= m["pr_auc"] <= 1 and 0 <= m["f1"] <= 1
    r = regression_metrics([1, 2, 3], [1, 2, 4])
    assert np.isclose(r["rmse"], np.sqrt(1 / 3)) and np.isclose(r["mae"], 1 / 3) and np.isclose(r["r2"], 0.5)
