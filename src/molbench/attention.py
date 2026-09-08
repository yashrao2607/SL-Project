"""Attention analysis for trained MAT models (PRD section 3.5).

Statistics computed per layer and head on the pure self-attention term (`self_attn`, what the model
learns) and on the fused attention (`attn`, what the model actually uses):
* bonded_share   - attention mass a real atom sends to its bonded neighbours
* self_share     - attention mass an atom sends to itself
* dummy_share    - attention mass sent to the dummy node
* dist_corr      - Spearman correlation between attention weight and inverse 3D distance (pairs of real atoms)
* entropy        - mean row entropy (nats) over real atoms
* received_by_class - mean attention received by atom classes (aromatic, heteroatom, ring, dummy)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from scipy.stats import spearmanr

from .engine import mol_collate

ATOM_CLASSES = ["aromatic", "heteroatom", "ring", "halogen", "carbon_aliphatic"]


def atom_classes(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    cls = {c: np.zeros(mol.GetNumAtoms(), dtype=bool) for c in ATOM_CLASSES}
    for a in mol.GetAtoms():
        i = a.GetIdx()
        cls["aromatic"][i] = a.GetIsAromatic()
        cls["heteroatom"][i] = a.GetAtomicNum() not in (1, 6)
        cls["ring"][i] = a.IsInRing()
        cls["halogen"][i] = a.GetAtomicNum() in (9, 17, 35, 53)
        cls["carbon_aliphatic"][i] = a.GetAtomicNum() == 6 and not a.GetIsAromatic()
    return cls


@torch.no_grad()
def collect_attention(model, recs, X_fp, smiles_list, batch_size: int = 32):
    """Run the model with attention capture and return per-molecule per-layer attention arrays."""
    model.eval()
    model.set_store_attention(True)
    out = []
    for s in range(0, len(recs), batch_size):
        chunk = list(zip(recs[s:s + batch_size], X_fp[s:s + batch_size], [0.0] * len(recs[s:s + batch_size])))
        b = mol_collate(chunk)
        model(b["afm"], b["mask"], b["adj"], b["dist"])
        maps = model.attention_maps()
        for i, rec in enumerate(recs[s:s + batch_size]):
            n = rec["afm"].shape[0]                      # includes dummy node
            out.append({
                "smiles": smiles_list[s + i], "n": n,
                "self_attn": [m["self_attn"][i, :, :n, :n].numpy() for m in maps],   # per layer (h, n, n)
                "attn": [m["attn"][i, :, :n, :n].numpy() for m in maps],
                "adj": rec["adj"][:n, :n], "dist": rec["dist"][:n, :n],
            })
    model.set_store_attention(False)
    return out


def _row_entropy(p):
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log(q)).sum(-1)


def head_statistics(items: list[dict]) -> pd.DataFrame:
    """Per (layer, head, kind) statistics averaged over molecules."""
    rows = []
    n_layers = len(items[0]["self_attn"])
    n_heads = items[0]["self_attn"][0].shape[0]
    for kind in ["self_attn", "attn"]:
        for layer in range(n_layers):
            for head in range(n_heads):
                bonded, selfs, dummy, corrs, ents = [], [], [], [], []
                for it in items:
                    n = it["n"]
                    if n < 4:
                        continue
                    P = it[kind][layer][head]                # (n, n) rows = queries
                    A = it["adj"].copy()
                    np.fill_diagonal(A, 0)
                    real = slice(1, n)
                    Pr = P[real, :]                           # real-atom queries
                    bonded.append((Pr[:, real] * A[real, real]).sum(-1).mean())
                    selfs.append(np.diag(P)[real].mean())
                    dummy.append(Pr[:, 0].mean())
                    ents.append(_row_entropy(Pr).mean())
                    D = it["dist"][real, real]
                    iu = np.triu_indices(n - 1, 1)
                    if len(iu[0]) >= 3:
                        w = (P[real, real][iu] + P[real, real].T[iu]) / 2
                        inv = 1.0 / np.maximum(D[iu], 1e-3)
                        c = spearmanr(w, inv).correlation
                        if not np.isnan(c):
                            corrs.append(c)
                rows.append({"kind": kind, "layer": layer, "head": head, "bonded_share": float(np.mean(bonded)),
                             "self_share": float(np.mean(selfs)), "dummy_share": float(np.mean(dummy)),
                             "dist_corr_spearman": float(np.mean(corrs)) if corrs else np.nan,
                             "entropy": float(np.mean(ents)), "n_molecules": len(bonded)})
    return pd.DataFrame(rows)


def received_by_class(items: list[dict]) -> pd.DataFrame:
    """Mean attention received per atom of each class (self-attention, averaged over heads), per layer."""
    rows = []
    n_layers = len(items[0]["self_attn"])
    for layer in range(n_layers):
        acc = {c: [] for c in ATOM_CLASSES + ["dummy"]}
        for it in items:
            n = it["n"]
            if n < 4:
                continue
            P = it["self_attn"][layer].mean(0)                # (n, n) averaged over heads
            received = P[1:, :].mean(0)                        # mean attention received from real-atom queries
            cls = atom_classes(it["smiles"])
            acc["dummy"].append(received[0])
            for c in ATOM_CLASSES:
                m = cls[c]
                if m.any():
                    acc[c].append(received[1:][m].mean())
        for c, v in acc.items():
            rows.append({"layer": layer, "atom_class": c, "mean_attention_received": float(np.mean(v)) if v else np.nan,
                         "n_molecules": len(v)})
    # normalise by the uniform baseline 1/n is molecule dependent, so we also report the ratio to the
    # average received by all real atoms
    return pd.DataFrame(rows)


def baseline_uniform_shares(items: list[dict]) -> dict:
    """What a uniform attention would give for bonded/self/dummy shares (for interpretation)."""
    b, s, d = [], [], []
    for it in items:
        n = it["n"]
        if n < 4:
            continue
        A = it["adj"].copy()
        np.fill_diagonal(A, 0)
        deg = A[1:, 1:].sum(-1)
        b.append((deg / n).mean())
        s.append(1.0 / n)
        d.append(1.0 / n)
    return {"bonded_share": float(np.mean(b)), "self_share": float(np.mean(s)), "dummy_share": float(np.mean(d))}
