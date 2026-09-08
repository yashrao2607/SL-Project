"""Fidelity of the MAT re-implementation against the authors' original transformer.py.

Given identical weights and identical inputs, the two implementations must produce identical
outputs (up to float32 round-off). The pretrained checkpoint must load with every encoder key covered.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from molbench import config as C
from molbench.models.mat import load_pretrained, make_model, make_pretrained_arch

ORIG_DIR = Path(__file__).resolve().parents[1] / "reference" / "mat_original"


def _original_make_model():
    sys.path.insert(0, str(ORIG_DIR))
    import importlib
    utils = importlib.import_module("utils")  # noqa: F841  (needed by transformer.py)
    transformer = importlib.import_module("transformer")
    return transformer.make_model


def _random_batch(B=3, L=12, d_atom=C.D_ATOM, seed=0):
    rng = np.random.RandomState(seed)
    afm = np.zeros((B, L, d_atom), dtype=np.float32)
    adj = np.zeros((B, L, L), dtype=np.float32)
    dist = np.zeros((B, L, L), dtype=np.float32)
    sizes = [L, L - 3, L - 6]
    for b, n in enumerate(sizes):
        afm[b, 0, 0] = 1.0                                    # dummy node
        afm[b, 1:n, 1:] = (rng.rand(n - 1, d_atom - 1) > 0.7).astype(np.float32)
        afm[b, 1:n, 1] = 1.0                                  # every real atom has a non-zero feature
        a = np.eye(n - 1, dtype=np.float32)
        for i in range(n - 2):
            a[i, i + 1] = a[i + 1, i] = 1.0
        adj[b, 1:n, 1:n] = a
        pos = rng.rand(n - 1, 3) * 5
        d = np.sqrt(((pos[:, None] - pos[None]) ** 2).sum(-1)).astype(np.float32)
        dd = np.full((n, n), 1e6, dtype=np.float32)
        dd[1:, 1:] = d
        dist[b, :n, :n] = dd
    afm_t = torch.from_numpy(afm)
    mask = afm_t.abs().sum(-1) != 0
    return afm_t, mask, torch.from_numpy(adj), torch.from_numpy(dist)


@pytest.mark.parametrize("kernel", ["softmax", "exp"])
@pytest.mark.parametrize("lambdas", [(0.33, 0.33), (0.5, 0.5), (0.5, 0.0), (0.0, 0.5)])
def test_forward_matches_original(kernel, lambdas):
    orig_make = _original_make_model()
    la, ld = lambdas
    common = dict(d_atom=C.D_ATOM, N=2, d_model=32, h=4, dropout=0.0, lambda_attention=la, lambda_distance=ld,
                  N_dense=1, leaky_relu_slope=0.1, dense_output_nonlinearity="relu", distance_matrix_kernel=kernel,
                  aggregation_type="mean")
    torch.manual_seed(1)
    mine = make_model(**common)
    theirs = orig_make(**common)
    theirs.load_state_dict(mine.state_dict())          # identical parameter names by construction
    mine.eval()
    theirs.eval()
    afm, mask, adj, dist = _random_batch()
    with torch.no_grad():
        out_mine = mine(afm, mask, adj, dist)
        out_theirs = theirs(afm, mask, adj, dist, None)
    assert out_mine.shape == out_theirs.shape == (3, 1)
    assert torch.allclose(out_mine, out_theirs, atol=1e-5, rtol=1e-5), (out_mine, out_theirs)


def test_pretrained_checkpoint_loads_completely():
    if not C.PRETRAINED_PATH.exists():
        pytest.skip("pretrained checkpoint not downloaded")
    model = make_pretrained_arch()
    info = load_pretrained(model)
    assert info["n_copied"] == 116            # 118 tensors minus generator.proj.{weight,bias}
    assert info["n_skipped_generator"] == 2
    assert info["missing_encoder_keys"] == []
    assert sum(p.numel() for p in model.parameters()) == 42077212 - (28 * 1024 + 28) + (1024 + 1)


def test_pretrained_forward_matches_original_with_checkpoint():
    if not C.PRETRAINED_PATH.exists():
        pytest.skip("pretrained checkpoint not downloaded")
    orig_make = _original_make_model()
    arch = dict(C.PRETRAINED_ARCH)
    mine = make_pretrained_arch()
    load_pretrained(mine)
    theirs = orig_make(d_atom=C.D_ATOM, **arch)
    theirs.load_state_dict(mine.state_dict())
    mine.eval()
    theirs.eval()
    afm, mask, adj, dist = _random_batch(seed=3)
    with torch.no_grad():
        a = mine(afm, mask, adj, dist)
        b = theirs(afm, mask, adj, dist, None)
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)
