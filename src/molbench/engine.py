"""Training and evaluation engine shared by every model family.

One call = one (model, task, split, seed, fold) run. Deep models are trained with Adam, early
stopping on the validation primary metric (ROC-AUC for classification, RMSE for regression) and
the best validation checkpoint is restored before the test set is scored.
"""
from __future__ import annotations

import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .metrics import compute_metrics
from .models.classical import fit_predict_rf, fit_predict_svm
from .models.gcn import GCN, record_to_pyg
from .models.hybrid import ECFPMATHybrid
from .models.mat import load_pretrained, make_mat_variant, make_pretrained_arch


DEVICE = torch.device(os.environ.get("MOLBENCH_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch):
    if isinstance(batch, dict):
        return {k: v.to(DEVICE) for k, v in batch.items()}
    return batch.to(DEVICE)


# --------------------------------------------------------------------------------------
# Batching for MAT-style models
# --------------------------------------------------------------------------------------
class MolDataset(Dataset):
    def __init__(self, recs, X_fp, y, indices):
        self.items = [(recs[i], X_fp[i], float(y[i])) for i in indices]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        return self.items[k]


def _pad2(a, L):
    out = np.zeros((L, L), dtype=np.float32)
    out[: a.shape[0], : a.shape[1]] = a
    return out


def mol_collate(batch):
    L = max(item[0]["afm"].shape[0] for item in batch)
    d = batch[0][0]["afm"].shape[1]
    afm = np.zeros((len(batch), L, d), dtype=np.float32)
    adj = np.zeros((len(batch), L, L), dtype=np.float32)
    dist = np.zeros((len(batch), L, L), dtype=np.float32)
    fp = np.zeros((len(batch), batch[0][1].shape[0]), dtype=np.float32)
    y = np.zeros(len(batch), dtype=np.float32)
    for b, (rec, f, yy) in enumerate(batch):
        n = rec["afm"].shape[0]
        afm[b, :n] = rec["afm"]
        adj[b] = _pad2(rec["adj"], L)
        dist[b] = _pad2(rec["dist"], L)
        fp[b] = f
        y[b] = yy
    afm_t = torch.from_numpy(afm)
    mask = (afm_t.abs().sum(dim=-1) != 0)
    return {"afm": afm_t, "adj": torch.from_numpy(adj), "dist": torch.from_numpy(dist), "mask": mask,
            "fp": torch.from_numpy(fp), "y": torch.from_numpy(y)}


class BucketBatchSampler:
    """Seeded 'sortish' batching: shuffle, sort by molecule size inside chunks of 20 batches, cut into
    batches, shuffle the batch order. Keeps SGD randomness while cutting padding waste on CPU."""

    def __init__(self, sizes, batch_size, seed, chunk_batches=20):
        self.sizes = np.asarray(sizes)
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.chunk = batch_size * chunk_batches

    def __iter__(self):
        perm = self.rng.permutation(len(self.sizes))
        batches = []
        for s in range(0, len(perm), self.chunk):
            chunk = perm[s:s + self.chunk]
            chunk = chunk[np.argsort(self.sizes[chunk], kind="stable")]
            batches += [chunk[i:i + self.batch_size].tolist() for i in range(0, len(chunk), self.batch_size)]
        order = self.rng.permutation(len(batches))
        for j in order:
            yield batches[j]

    def __len__(self):
        return int(np.ceil(len(self.sizes) / self.batch_size))


def make_loader(recs, X_fp, y, indices, batch_size, shuffle, seed):
    ds = MolDataset(recs, X_fp, y, indices)
    if shuffle:
        sampler = BucketBatchSampler([recs[i]["afm"].shape[0] for i in indices], batch_size, seed)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=mol_collate)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=mol_collate)


def make_pyg_loader(recs, y, indices, batch_size, shuffle, seed):
    from torch_geometric.loader import DataLoader as PyGLoader
    data = [record_to_pyg(recs[i], float(y[i])) for i in indices]
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return PyGLoader(data, batch_size=batch_size, shuffle=shuffle, generator=gen)


# --------------------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------------------
def build_torch_model(model_name: str, cfg: dict) -> nn.Module:
    if model_name == "gcn":
        return GCN(in_dim=C.D_ATOM_BASE, hidden=cfg["hidden"], n_layers=cfg["n_layers"], dropout=cfg["dropout"])
    if model_name in ("mat", "mat_nograph", "mat_nodistance", "mat_noattention"):
        return make_mat_variant(model_name, d_model=cfg["d_model"], N=cfg["N"], h=cfg["h"], dropout=cfg["dropout"],
                                distance_matrix_kernel=cfg.get("distance_matrix_kernel", "softmax"))
    if model_name == "hybrid":
        mat = make_mat_variant("mat", d_model=cfg["d_model"], N=cfg["N"], h=cfg["h"], dropout=cfg["dropout"],
                               distance_matrix_kernel=cfg.get("distance_matrix_kernel", "softmax"))
        return ECFPMATHybrid(mat, d_model=cfg["d_model"], n_bits=C.ECFP_BITS, d_fp=cfg.get("d_fp", 128),
                             dropout=cfg["dropout"])
    if model_name == "mat_large_pretrained":
        model = make_pretrained_arch(dropout=cfg.get("dropout", 0.0))
        info = load_pretrained(model)
        assert info["n_copied"] > 0 and not info["missing_encoder_keys"]
        return model
    if model_name == "mat_large_scratch":
        return make_pretrained_arch(dropout=cfg.get("dropout", 0.0))
    raise ValueError(model_name)


def forward(model, model_name, batch):
    batch = to_device(batch)
    if model_name == "gcn":
        return model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
    if model_name == "hybrid":
        return model(batch["afm"], batch["mask"], batch["adj"], batch["dist"], batch["fp"]).squeeze(-1)
    return model(batch["afm"], batch["mask"], batch["adj"], batch["dist"]).squeeze(-1)


def batch_targets(model_name, batch):
    return batch.y.view(-1) if model_name == "gcn" else batch["y"]


@torch.no_grad()
def predict(model, model_name, loader, task_type, y_mu=0.0, y_sd=1.0):
    model.eval()
    outs, ys = [], []
    for batch in loader:
        out = forward(model, model_name, batch)
        outs.append(out.cpu().numpy())
        ys.append(batch_targets(model_name, batch).cpu().numpy())
    pred = np.concatenate(outs)
    y = np.concatenate(ys)
    if task_type == "clf":
        pred = 1.0 / (1.0 + np.exp(-pred))
    else:
        pred = pred * y_sd + y_mu
    return y, pred


def _primary(metrics: dict, task_type: str) -> float:
    m = C.PRIMARY_METRIC[task_type]
    v = metrics[m]
    return v if C.HIGHER_IS_BETTER[m] else -v      # always "higher is better" for early stopping


def run_torch(model_name: str, cfg: dict, recs, X_fp, y, task_type: str, fold: dict, seed: int,
              max_epochs: int = 100, patience: int = 15, n_threads: int = 2, return_model: bool = False,
              verbose: bool = False) -> dict:
    torch.set_num_threads(n_threads)
    seed_everything(seed)
    tr, va, te = fold["train"], fold["val"], fold["test"]
    y = np.asarray(y, dtype=np.float32)
    # Regression targets are standardised with training statistics and inverted for scoring.
    y_mu, y_sd = (float(y[tr].mean()), float(y[tr].std() + 1e-8)) if task_type == "reg" else (0.0, 1.0)
    y_model = (y - y_mu) / y_sd if task_type == "reg" else y
    bs = cfg.get("batch_size", 32)
    if model_name == "gcn":
        train_loader = make_pyg_loader(recs, y_model, tr, bs, True, seed)
        val_loader = make_pyg_loader(recs, y_model, va, 128, False, seed)
        test_loader = make_pyg_loader(recs, y_model, te, 128, False, seed)
    else:
        train_loader = make_loader(recs, X_fp, y_model, tr, bs, True, seed)
        val_loader = make_loader(recs, X_fp, y_model, va, 128, False, seed)
        test_loader = make_loader(recs, X_fp, y_model, te, 128, False, seed)

    model = build_torch_model(model_name, cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = nn.BCEWithLogitsLoss() if task_type == "clf" else nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))

    best_score, best_state, best_epoch, bad = -np.inf, None, -1, 0
    t0 = time.time()
    history = []
    for epoch in range(max_epochs):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            opt.zero_grad()
            out = forward(model, model_name, batch)
            loss = loss_fn(out, batch_targets(model_name, batch).to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * out.shape[0]
            count += out.shape[0]
        yv, pv = predict(model, model_name, val_loader, task_type, y_mu, y_sd)
        yv_orig = yv * y_sd + y_mu if task_type == "reg" else yv
        if not np.all(np.isfinite(pv)):                 # diverged epoch: never select it, count towards patience
            history.append({"epoch": epoch, "train_loss": total / max(count, 1), "diverged": True})
            bad += 1
            if bad >= patience:
                break
            continue
        vm = compute_metrics(task_type, yv_orig, pv)
        score = _primary(vm, task_type)
        if np.isnan(score):                              # single-class validation set: fall back to Brier / MSE
            score = -float(np.mean((pv - yv_orig) ** 2))
        history.append({"epoch": epoch, "train_loss": total / max(count, 1), **{f"val_{k}": v for k, v in vm.items()}})
        if verbose:
            print(f"  epoch {epoch:3d} loss {total / max(count, 1):.4f} val {C.PRIMARY_METRIC[task_type]} {vm[C.PRIMARY_METRIC[task_type]]:.4f}")
        if score > best_score + 1e-6:
            best_score, best_epoch, bad = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"{model_name}: no epoch produced finite validation predictions")
    model.load_state_dict(best_state)
    model.to(DEVICE)
    yv, pv = predict(model, model_name, val_loader, task_type, y_mu, y_sd)
    yt, pt = predict(model, model_name, test_loader, task_type, y_mu, y_sd)
    if task_type == "reg":
        yv, yt = yv * y_sd + y_mu, yt * y_sd + y_mu
    val_metrics = compute_metrics(task_type, yv, pv)
    test_metrics = compute_metrics(task_type, yt, pt)
    result = {**test_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}, "best_epoch": best_epoch,
              "epochs_run": len(history), "n_params": n_params, "wall_time_s": round(time.time() - t0, 1)}
    if return_model:
        result["model"] = model
        result["history"] = history
        result["test_pred"] = pt
        result["test_true"] = yt
    return result


def run_classical(model_name: str, cfg: dict, X_fp, y, task_type: str, fold: dict, seed: int, n_jobs: int = 1) -> dict:
    t0 = time.time()
    tr, va, te = fold["train"], fold["val"], fold["test"]
    y = np.asarray(y, dtype=np.float32)
    if model_name == "rf":
        pv, pt = fit_predict_rf(X_fp[tr], y[tr], [X_fp[va], X_fp[te]], task_type, cfg, seed, n_jobs=n_jobs)
    elif model_name == "svm":
        pv, pt = fit_predict_svm(X_fp[tr], y[tr], [X_fp[va], X_fp[te]], task_type, cfg, seed)
    else:
        raise ValueError(model_name)
    val_metrics = compute_metrics(task_type, y[va], pv)
    test_metrics = compute_metrics(task_type, y[te], pt)
    return {**test_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}, "best_epoch": -1, "epochs_run": 0,
            "n_params": 0, "wall_time_s": round(time.time() - t0, 1)}


def run_one(model_name: str, cfg: dict, recs, X_fp, y, task_type: str, fold: dict, seed: int, **kw) -> dict:
    if model_name in ("rf", "svm"):
        return run_classical(model_name, cfg, X_fp, y, task_type, fold, seed, n_jobs=kw.get("n_threads", 1))
    return run_torch(model_name, cfg, recs, X_fp, y, task_type, fold, seed, **kw)
