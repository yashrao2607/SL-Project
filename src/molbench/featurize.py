"""Featurisation: ECFP fingerprints, atom features, ETKDG conformers, adjacency and distance matrices.

The atom featurisation and the (node features, adjacency, distance) triple follow the original
MAT code (`reference/mat_original/data_utils.py`) exactly, with one-hot formal charge enabled so
that the released pretrained weights (d_atom = 28) can be loaded.
"""
from __future__ import annotations

import pickle
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFingerprintGenerator
from sklearn.metrics import pairwise_distances

from . import config as C

RDLogger.DisableLog("rdApp.*")

_ATOMIC_NUMS = [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
_DEGREES = [0, 1, 2, 3, 4, 5]
_NUM_HS = [0, 1, 2, 3, 4]
_CHARGES = [-1, 0, 1]


def _one_hot(val, options):
    if val not in options:
        val = options[-1]
    return [float(x == val) for x in options]


def atom_features(atom: Chem.Atom, one_hot_formal_charge: bool = C.ONE_HOT_FORMAL_CHARGE) -> np.ndarray:
    feats = []
    feats += _one_hot(atom.GetAtomicNum(), _ATOMIC_NUMS)
    feats += _one_hot(len(atom.GetNeighbors()), _DEGREES)
    feats += _one_hot(atom.GetTotalNumHs(), _NUM_HS)
    if one_hot_formal_charge:
        feats += _one_hot(atom.GetFormalCharge(), _CHARGES)
    else:
        feats.append(float(atom.GetFormalCharge()))
    feats.append(float(atom.IsInRing()))
    feats.append(float(atom.GetIsAromatic()))
    return np.array(feats, dtype=np.float32)


# --------------------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------------------
_FP_GEN = None


def _fp_generator():
    global _FP_GEN
    if _FP_GEN is None:
        _FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=C.ECFP_RADIUS, fpSize=C.ECFP_BITS)
    return _FP_GEN


def ecfp(smiles: str) -> np.ndarray:
    """ECFP4-style Morgan fingerprint (radius 2, 2048 bits) as uint8 vector."""
    mol = Chem.MolFromSmiles(smiles)
    return _fp_generator().GetFingerprintAsNumPy(mol).astype(np.uint8)


# --------------------------------------------------------------------------------------
# 3D conformers (ETKDG + UFF) with the original 2D fallback
# --------------------------------------------------------------------------------------
def embed_conformer(mol: Chem.Mol, seed: int = C.CONFORMER_SEED):
    """Return (mol_with_conformer, method) where method is 'etkdg' or '2d'."""
    molh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    # Bounded effort: RDKit's default iteration budget (10 x n_atoms), then one retry with random
    # initial coordinates (RDKit's recommendation for large / macrocyclic molecules), then 2D.
    if mol.GetNumHeavyAtoms() > 60:
        params.useRandomCoords = True
    cid = AllChem.EmbedMolecule(molh, params)
    if cid < 0 and not params.useRandomCoords:
        params.useRandomCoords = True
        cid = AllChem.EmbedMolecule(molh, params)
    if cid >= 0:
        try:
            AllChem.UFFOptimizeMolecule(molh, maxIters=200)
        except Exception:
            pass
        out = Chem.RemoveHs(molh)
        return out, "etkdg"
    out = Chem.Mol(mol)
    AllChem.Compute2DCoords(out)
    return out, "2d"


def featurize_mol(mol: Chem.Mol, add_dummy_node: bool = True):
    """Return (node_features, adjacency, distance) exactly as the original MAT featuriser."""
    node_features = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    n = mol.GetNumAtoms()
    adj = np.eye(n, dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i, j] = adj[j, i] = 1.0
    conf = mol.GetConformer()
    pos = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                    for k in range(n)], dtype=np.float64)
    dist = pairwise_distances(pos).astype(np.float32)
    if add_dummy_node:
        m = np.zeros((n + 1, node_features.shape[1] + 1), dtype=np.float32)
        m[1:, 1:] = node_features
        m[0, 0] = 1.0
        node_features = m
        a = np.zeros((n + 1, n + 1), dtype=np.float32)
        a[1:, 1:] = adj
        adj = a
        d = np.full((n + 1, n + 1), 1e6, dtype=np.float32)
        d[1:, 1:] = dist
        dist = d
    return node_features, adj, dist


def featurize_smiles(smiles: str) -> dict:
    """Full feature record for one canonical SMILES (used by the multiprocessing pool).

    Never raises for a parsable molecule: if 3D embedding throws, the original code's 2D fallback is
    used and the record is tagged `conformer="2d"`.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES: {smiles}")
    try:
        mol3d, method = embed_conformer(mol)
    except Exception as e:  # pragma: no cover - RDKit embedding failure on exotic molecules
        mol3d = Chem.Mol(mol)
        AllChem.Compute2DCoords(mol3d)
        method = "2d"
    afm, adj, dist = featurize_mol(mol3d, add_dummy_node=True)
    return {"smiles": smiles, "afm": afm, "adj": adj, "dist": dist, "ecfp": ecfp(smiles),
            "conformer": method, "n_atoms": int(mol.GetNumAtoms())}


def cache_path():
    return C.FEATURE_DIR / "mol_cache.pkl"


def load_cache() -> dict:
    p = cache_path()
    if p.exists():
        with open(p, "rb") as fh:
            return pickle.load(fh)
    return {}


def save_cache(cache: dict) -> None:
    with open(cache_path(), "wb") as fh:
        pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)


def featurize_all(smiles_list: list[str], n_jobs: int = 8, chunksize: int = 1, verbose: bool = True) -> dict:
    """Featurise every SMILES not yet in the on-disk cache; return the full cache."""
    cache = load_cache()
    todo = sorted({s for s in smiles_list if s not in cache}, key=lambda x: (-len(x), x))   # largest first
    if verbose:
        print(f"[featurize] cached={len(cache)} todo={len(todo)} n_jobs={n_jobs}")
    if todo:
        if n_jobs > 1:
            with Pool(n_jobs) as pool:
                for i, rec in enumerate(pool.imap_unordered(featurize_smiles, todo, chunksize=chunksize)):
                    cache[rec["smiles"]] = rec
                    if verbose and (i + 1) % 500 == 0:
                        print(f"[featurize] {i + 1}/{len(todo)}", flush=True)
        else:
            for i, s in enumerate(todo):
                cache[s] = featurize_smiles(s)
        save_cache(cache)
    return cache


def task_features(task: str, cache: dict | None = None):
    """Return (records list aligned with the processed CSV, ecfp matrix, y, task_type)."""
    from .data import load_processed
    df = load_processed(task)
    cache = cache if cache is not None else load_cache()
    recs = [cache[s] for s in df["smiles"]]
    X_fp = np.stack([r["ecfp"] for r in recs]).astype(np.float32)
    y = df["y"].values.astype(np.float32)
    return recs, X_fp, y, C.TASKS[task]["type"]
