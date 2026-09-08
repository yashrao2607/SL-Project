"""Classical QSAR baselines: Random Forest and Tanimoto-kernel SVM / SVR on ECFP4 fingerprints."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR


def tanimoto_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tanimoto (Jaccard) similarity between rows of binary matrices A (n x d) and B (m x d)."""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    inter = A @ B.T
    na = A.sum(axis=1, keepdims=True)
    nb = B.sum(axis=1, keepdims=True).T
    denom = na + nb - inter
    return np.where(denom > 0, inter / np.maximum(denom, 1e-9), 0.0)


def fit_predict_rf(X_train, y_train, X_eval_list, task_type: str, params: dict, seed: int, n_jobs: int = 1):
    p = dict(params)
    if task_type == "clf":
        model = RandomForestClassifier(n_estimators=p.pop("n_estimators", 500), random_state=seed, n_jobs=n_jobs, **p)
        model.fit(X_train, y_train.astype(int))
        return [model.predict_proba(X)[:, 1] for X in X_eval_list]
    p.pop("class_weight", None)
    model = RandomForestRegressor(n_estimators=p.pop("n_estimators", 500), random_state=seed, n_jobs=n_jobs, **p)
    model.fit(X_train, y_train)
    return [model.predict(X) for X in X_eval_list]


def fit_predict_svm(X_train, y_train, X_eval_list, task_type: str, params: dict, seed: int):
    """SVM/SVR with a precomputed Tanimoto kernel. Classification scores are sigmoid(decision)."""
    K_train = tanimoto_kernel(X_train, X_train)
    K_evals = [tanimoto_kernel(X, X_train) for X in X_eval_list]
    p = dict(params)
    if task_type == "clf":
        model = SVC(kernel="precomputed", C=p.get("C", 1.0), class_weight=p.get("class_weight"), random_state=seed)
        model.fit(K_train, y_train.astype(int))
        return [1.0 / (1.0 + np.exp(-model.decision_function(K))) for K in K_evals]
    model = SVR(kernel="precomputed", C=p.get("C", 1.0), epsilon=p.get("epsilon", 0.1))
    model.fit(K_train, y_train)
    return [model.predict(K) for K in K_evals]
