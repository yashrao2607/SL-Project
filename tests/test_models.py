"""Unit tests for GCN, hybrid, MAT ablations and attention capture."""
import numpy as np
import torch

from molbench import config as C
from molbench.models.gcn import GCN, record_to_pyg
from molbench.models.hybrid import ECFPMATHybrid
from molbench.models.mat import ABLATION_LAMBDAS, make_mat_variant
from molbench.featurize import featurize_smiles
from molbench.engine import mol_collate


def _batch(smiles=("CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1")):
    recs = [featurize_smiles(s) for s in smiles]
    fps = [r["ecfp"].astype(np.float32) for r in recs]
    return recs, mol_collate([(r, f, 0.0) for r, f in zip(recs, fps)])


def test_ablation_lambdas_sum_to_one():
    for v, (a, d, g) in ABLATION_LAMBDAS.items():
        assert abs(a + d + g - 1) < 1e-9
        m = make_mat_variant(v)
        assert m.lambdas == (a, d, g)


def test_nograph_is_invariant_to_adjacency():
    torch.manual_seed(0)
    _, b = _batch()
    m = make_mat_variant("mat_nograph", dropout=0.0).eval()
    with torch.no_grad():
        out1 = m(b["afm"], b["mask"], b["adj"], b["dist"])
        adj2 = torch.zeros_like(b["adj"]) + torch.eye(b["adj"].shape[1])
        out2 = m(b["afm"], b["mask"], adj2, b["dist"])
    assert torch.allclose(out1, out2, atol=1e-6)


def test_nodistance_is_invariant_to_distances():
    torch.manual_seed(0)
    _, b = _batch()
    m = make_mat_variant("mat_nodistance", dropout=0.0).eval()
    with torch.no_grad():
        out1 = m(b["afm"], b["mask"], b["adj"], b["dist"])
        out2 = m(b["afm"], b["mask"], b["adj"], b["dist"] * 3.0 + 1.0)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_noattention_gives_no_gradient_to_query_key():
    torch.manual_seed(0)
    _, b = _batch()
    m = make_mat_variant("mat_noattention", dropout=0.0)
    out = m(b["afm"], b["mask"], b["adj"], b["dist"]).sum()
    out.backward()
    for layer in m.encoder.layers:
        q_grad = layer.self_attn.linears[0].weight.grad
        k_grad = layer.self_attn.linears[1].weight.grad
        v_grad = layer.self_attn.linears[2].weight.grad
        assert q_grad is None or torch.all(q_grad == 0)
        assert k_grad is None or torch.all(k_grad == 0)
        assert v_grad is not None and torch.any(v_grad != 0)


def test_full_mat_uses_all_three_terms():
    torch.manual_seed(0)
    _, b = _batch()
    m = make_mat_variant("mat", dropout=0.0).eval()
    with torch.no_grad():
        base = m(b["afm"], b["mask"], b["adj"], b["dist"])
        no_adj = m(b["afm"], b["mask"], torch.eye(b["adj"].shape[1]).expand_as(b["adj"]).clone(), b["dist"])
        far = m(b["afm"], b["mask"], b["adj"], b["dist"] * 3.0 + 1.0)
    assert not torch.allclose(base, no_adj, atol=1e-6)
    assert not torch.allclose(base, far, atol=1e-6)


def test_attention_capture_rows_sum_to_one():
    torch.manual_seed(0)
    _, b = _batch()
    m = make_mat_variant("mat", dropout=0.0).eval()
    m.set_store_attention(True)
    with torch.no_grad():
        m(b["afm"], b["mask"], b["adj"], b["dist"])
    maps = m.attention_maps()
    assert len(maps) == 2
    p = maps[0]["self_attn"]                         # (B, h, L, L)
    assert p.shape[1] == 4
    real = b["mask"]
    for i in range(p.shape[0]):
        n = int(real[i].sum())
        rows = p[i, :, :n, :n].sum(-1)
        assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)
    fused = maps[0]["attn"]
    n = int(real[0].sum())
    # fused rows of real (non-dummy) atoms sum to ~1 (softmax kernel + normalised adjacency + attention)
    assert torch.allclose(fused[0, :, 1:n, :n].sum(-1), torch.ones(4, n - 1), atol=1e-4)


def test_padding_does_not_change_outputs():
    torch.manual_seed(0)
    recs, b = _batch()
    m = make_mat_variant("mat", dropout=0.0).eval()
    single = mol_collate([(recs[0], recs[0]["ecfp"].astype(np.float32), 0.0)])
    with torch.no_grad():
        batched = m(b["afm"], b["mask"], b["adj"], b["dist"])[0]
        alone = m(single["afm"], single["mask"], single["adj"], single["dist"])[0]
    assert torch.allclose(batched, alone, atol=1e-5)


def test_gcn_forward_and_gradients():
    from torch_geometric.loader import DataLoader
    recs, _ = _batch()
    data = [record_to_pyg(r, 1.0) for r in recs]
    batch = next(iter(DataLoader(data, batch_size=3)))
    assert batch.x.shape[1] == C.D_ATOM_BASE
    m = GCN(in_dim=C.D_ATOM_BASE, hidden=16, n_layers=2)
    out = m(batch.x, batch.edge_index, batch.batch)
    assert out.shape == (3, 1)
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters())


def test_single_atom_molecule_gcn():
    rec = featurize_smiles("C")
    d = record_to_pyg(rec, 0.0)
    assert d.x.shape[0] == 1 and d.edge_index.shape == (2, 0)


def test_hybrid_forward():
    _, b = _batch()
    mat = make_mat_variant("mat", d_model=32, N=1, h=4)
    hy = ECFPMATHybrid(mat, d_model=32, d_fp=16)
    out = hy(b["afm"], b["mask"], b["adj"], b["dist"], b["fp"])
    assert out.shape == (3, 1)
    out.sum().backward()
    assert hy.fp_proj[0].weight.grad is not None
    assert mat.src_embed.lut.weight.grad is not None
