"""End-to-end engine tests on a tiny synthetic dataset (no feature cache required)."""
import numpy as np
import pytest

from molbench.engine import run_classical, run_one, run_torch
from molbench.featurize import featurize_smiles

SMILES = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "c1ccccc1O", "c1ccccc1N", "CC(=O)O", "CCC(=O)O", "CC(=O)N",
          "CCN", "CCCN", "c1ccncc1", "c1ccncc1C", "CCOC", "CCOCC", "C1CCCCC1", "C1CCCC1", "CC(C)O",
          "CC(C)(C)O", "CCS", "CCCS", "c1ccsc1", "c1ccoc1", "CC=O"]


@pytest.fixture(scope="module")
def tiny():
    recs = [featurize_smiles(s) for s in SMILES]
    X_fp = np.stack([r["ecfp"] for r in recs]).astype(np.float32)
    rng = np.random.RandomState(0)
    y_clf = np.array([1.0 if "c1" in s else 0.0 for s in SMILES], dtype=np.float32)
    y_reg = np.array([len(s) + rng.randn() * 0.1 for s in SMILES], dtype=np.float32)
    n = len(SMILES)
    idx = list(range(n))
    fold = {"train": idx[:16], "val": [16, 17, 21, 22], "test": [18, 19, 20, 23]}   # val and test both contain both classes
    return recs, X_fp, y_clf, y_reg, fold


@pytest.mark.parametrize("model", ["mat", "mat_nograph", "mat_nodistance", "mat_noattention", "hybrid", "gcn"])
def test_torch_models_train_clf_and_reg(tiny, model):
    recs, X_fp, y_clf, y_reg, fold = tiny
    cfg = {"d_model": 32, "N": 1, "h": 4, "lr": 1e-3, "dropout": 0.1, "batch_size": 8, "hidden": 16, "n_layers": 2}
    r = run_torch(model, cfg, recs, X_fp, y_clf, "clf", fold, seed=42, max_epochs=3, patience=2, n_threads=2)
    assert set(["roc_auc", "pr_auc", "f1", "val_roc_auc", "best_epoch", "epochs_run", "wall_time_s"]) <= set(r)
    assert 0 <= r["f1"] <= 1 and r["epochs_run"] <= 3
    r2 = run_torch(model, cfg, recs, X_fp, y_reg, "reg", fold, seed=42, max_epochs=3, patience=2, n_threads=2)
    assert r2["rmse"] > 0 and np.isfinite(r2["mae"]) and np.isfinite(r2["r2"])
    # RMSE must be on the original label scale (labels ~ 3..12, std ~ 2.5): a standardised-scale bug would give ~1
    assert r2["rmse"] < 20


def test_regression_metrics_are_in_original_units(tiny):
    recs, X_fp, _, y_reg, fold = tiny
    cfg = {"d_model": 16, "N": 1, "h": 4, "lr": 1e-3, "dropout": 0.0, "batch_size": 8}
    r = run_torch("mat", cfg, recs, X_fp, y_reg, "reg", fold, seed=1, max_epochs=1, patience=1, n_threads=2,
                  return_model=True)
    assert r["test_true"].shape == (len(fold["test"]),)
    assert np.allclose(np.sort(r["test_true"]), np.sort(y_reg[fold["test"]]), atol=1e-5)


def test_classical_models(tiny):
    recs, X_fp, y_clf, y_reg, fold = tiny
    for model, cfg in [("rf", {"n_estimators": 50, "max_features": "sqrt", "min_samples_leaf": 1, "class_weight": None}),
                       ("svm", {"C": 1.0, "class_weight": None})]:
        r = run_classical(model, cfg, X_fp, y_clf, "clf", fold, seed=42)
        assert 0 <= r["roc_auc"] <= 1 and 0 <= r["f1"] <= 1
    for model, cfg in [("rf", {"n_estimators": 50, "max_features": "sqrt", "min_samples_leaf": 1}),
                       ("svm", {"C": 1.0, "epsilon": 0.1})]:
        r = run_classical(model, cfg, X_fp, y_reg, "reg", fold, seed=42)
        assert r["rmse"] > 0


def test_determinism_same_seed_same_result(tiny):
    recs, X_fp, y_clf, _, fold = tiny
    cfg = {"d_model": 16, "N": 1, "h": 4, "lr": 1e-3, "dropout": 0.1, "batch_size": 8}
    a = run_one("mat", cfg, recs, X_fp, y_clf, "clf", fold, 7, max_epochs=2, patience=2, n_threads=2)
    b = run_one("mat", cfg, recs, X_fp, y_clf, "clf", fold, 7, max_epochs=2, patience=2, n_threads=2)
    assert a["roc_auc"] == b["roc_auc"] and a["val_roc_auc"] == b["val_roc_auc"]
    assert not np.isnan(a["val_roc_auc"]) and not np.isnan(a["roc_auc"])
