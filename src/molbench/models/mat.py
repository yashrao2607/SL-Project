"""Molecule Attention Transformer (MAT) - faithful PyTorch re-implementation.

Reference: Maziarka et al., "Molecule Attention Transformer", arXiv:2002.08264, and the authors'
code (`reference/mat_original/transformer.py`). Parameter names are kept identical to the original
so that (a) the released pretrained checkpoint loads directly and (b) `tests/test_mat_fidelity.py`
can verify bit-for-bit equivalence of the forward pass against the original implementation.

Extensions over the original:
* `lambda_adjacency` can be set explicitly, which gives the three ablations
  MAT-NoGraph (lambda_adjacency = 0), MAT-NoDistance (lambda_distance = 0) and
  MAT-NoAttention (lambda_attention = 0).
* `store_attention` keeps the per-head self-attention (`p_attn`) and the fused attention
  (`p_weighted`) of every layer for the attention analysis.
* `pool()` exposes the masked mean pooled embedding (used by the ECFP+MAT hybrid).
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config as C


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class LayerNorm(nn.Module):
    """LayerNorm exactly as in the original code (unbiased std, eps added to std)."""

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """Pre-norm residual block: x + dropout(sublayer(norm(x)))."""

    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadedAttention(nn.Module):
    """Molecule self-attention: lambda_a * softmax(QK^T/sqrt(d)) + lambda_d * g(D) + lambda_g * A_norm."""

    def __init__(self, h: int, d_model: int, dropout: float = 0.1, lambda_attention: float = 0.33,
                 lambda_distance: float = 0.33, lambda_adjacency: float | None = None,
                 distance_matrix_kernel: str = "softmax"):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by the number of heads"
        self.d_k = d_model // h
        self.h = h
        if lambda_adjacency is None:
            lambda_adjacency = 1.0 - lambda_attention - lambda_distance
        self.lambdas = (float(lambda_attention), float(lambda_distance), float(lambda_adjacency))
        assert abs(sum(self.lambdas) - 1.0) < 1e-6, f"lambdas must sum to 1, got {self.lambdas}"
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        if distance_matrix_kernel == "softmax":
            self.distance_matrix_kernel = lambda x: F.softmax(-x, dim=-1)
        elif distance_matrix_kernel == "exp":
            self.distance_matrix_kernel = lambda x: torch.exp(-x)
        else:
            raise ValueError(distance_matrix_kernel)
        self.store_attention = False
        self.attn = None        # fused attention p_weighted   (B, h, L, L)
        self.self_attn = None   # pure self-attention p_attn   (B, h, L, L)
        self.p_dist = None      # distance term                (B, L, L)
        self.p_adj = None       # adjacency term               (B, L, L)

    def forward(self, query, key, value, adj_matrix, distances_matrix, mask):
        # mask: (B, L) bool/0-1 with True for real atoms (dummy node included).
        nbatches, L = query.size(0), query.size(1)
        mask3 = mask.unsqueeze(1)                                   # (B, 1, L)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # Distance term: mask padded key columns with +inf, then kernel.
        dist = distances_matrix.masked_fill(mask3.expand(-1, L, -1) == 0, float("inf"))
        p_dist = self.distance_matrix_kernel(dist)                  # (B, L, L)
        # Self-attention term.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask3.unsqueeze(1) == 0, -1e12)
        p_attn = F.softmax(scores, dim=-1)                          # (B, h, L, L)
        # Graph term: row-normalised adjacency (self-loops included in the input matrix).
        p_adj = adj_matrix / (adj_matrix.sum(dim=-1, keepdim=True) + 1e-6)
        la, ld, lg = self.lambdas
        p_weighted = la * p_attn + ld * p_dist.unsqueeze(1) + lg * p_adj.unsqueeze(1)
        if self.store_attention:
            self.attn = p_weighted.detach()
            self.self_attn = p_attn.detach()
            self.p_dist = p_dist.detach()
            self.p_adj = p_adj.detach()
        p_weighted = self.dropout(p_weighted)
        x = torch.matmul(p_weighted, value)                         # (B, h, L, d_k)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, N_dense: int, dropout: float = 0.1, leaky_relu_slope: float = 0.0,
                 dense_output_nonlinearity: str = "relu"):
        super().__init__()
        self.N_dense = N_dense
        self.linears = clones(nn.Linear(d_model, d_model), N_dense)
        self.dropout = clones(nn.Dropout(dropout), N_dense)
        self.leaky_relu_slope = leaky_relu_slope
        if dense_output_nonlinearity == "relu":
            self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        elif dense_output_nonlinearity == "tanh":
            self.dense_output_nonlinearity = torch.tanh
        elif dense_output_nonlinearity == "none":
            self.dense_output_nonlinearity = lambda x: x
        else:
            raise ValueError(dense_output_nonlinearity)

    def forward(self, x):
        if self.N_dense == 0:
            return x
        for i in range(len(self.linears) - 1):
            x = self.dropout[i](F.leaky_relu(self.linears[i](x), negative_slope=self.leaky_relu_slope))
        return self.dropout[-1](self.dense_output_nonlinearity(self.linears[-1](x)))


class EncoderLayer(nn.Module):
    def __init__(self, size: int, self_attn: MultiHeadedAttention, feed_forward: PositionwiseFeedForward,
                 dropout: float):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, adj_matrix, distances_matrix):
        x = self.sublayer[0](x, lambda t: self.self_attn(t, t, t, adj_matrix, distances_matrix, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, adj_matrix, distances_matrix):
        for layer in self.layers:
            x = layer(x, mask, adj_matrix, distances_matrix)
        return self.norm(x)


class Embeddings(nn.Module):
    def __init__(self, d_model: int, d_atom: int, dropout: float):
        super().__init__()
        self.lut = nn.Linear(d_atom, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.lut(x))


def masked_mean_pool(x, mask):
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(dim=1) / m.sum(dim=1)


class Generator(nn.Module):
    def __init__(self, d_model: int, aggregation_type: str = "mean", n_output: int = 1, n_layers: int = 1,
                 leaky_relu_slope: float = 0.01, dropout: float = 0.0):
        super().__init__()
        if n_layers == 1:
            self.proj = nn.Linear(d_model, n_output)
        else:
            layers = []
            for _ in range(n_layers - 1):
                layers += [nn.Linear(d_model, d_model), nn.LeakyReLU(leaky_relu_slope), LayerNorm(d_model),
                           nn.Dropout(dropout)]
            layers.append(nn.Linear(d_model, n_output))
            self.proj = nn.Sequential(*layers)
        self.aggregation_type = aggregation_type

    def forward(self, x, mask):
        m = mask.unsqueeze(-1).float()
        out_masked = x * m
        if self.aggregation_type == "mean":
            pooled = out_masked.sum(dim=1) / m.sum(dim=1)
        elif self.aggregation_type == "sum":
            pooled = out_masked.sum(dim=1)
        elif self.aggregation_type == "dummy_node":
            pooled = out_masked[:, 0]
        else:
            raise ValueError(self.aggregation_type)
        return self.proj(pooled)


class GraphTransformer(nn.Module):
    def __init__(self, encoder: Encoder, src_embed: Embeddings, generator: Generator):
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.generator = generator

    def forward(self, src, src_mask, adj_matrix, distances_matrix):
        return self.predict(self.encode(src, src_mask, adj_matrix, distances_matrix), src_mask)

    def encode(self, src, src_mask, adj_matrix, distances_matrix):
        return self.encoder(self.src_embed(src), src_mask, adj_matrix, distances_matrix)

    def predict(self, out, out_mask):
        return self.generator(out, out_mask)

    def pool(self, src, src_mask, adj_matrix, distances_matrix):
        return masked_mean_pool(self.encode(src, src_mask, adj_matrix, distances_matrix), src_mask)

    # ---- attention utilities ----
    def set_store_attention(self, flag: bool = True):
        for layer in self.encoder.layers:
            layer.self_attn.store_attention = flag

    def attention_maps(self):
        """List over layers of dicts with fused (`attn`), self-attention (`self_attn`), distance and graph terms."""
        return [{"attn": l.self_attn.attn, "self_attn": l.self_attn.self_attn, "p_dist": l.self_attn.p_dist,
                 "p_adj": l.self_attn.p_adj} for l in self.encoder.layers]

    @property
    def lambdas(self):
        return self.encoder.layers[0].self_attn.lambdas


# --------------------------------------------------------------------------------------
# Factory, ablations, pretrained loading
# --------------------------------------------------------------------------------------
ABLATION_LAMBDAS = {
    "mat":             (0.33, 0.33, 0.34),
    "mat_nograph":     (0.50, 0.50, 0.00),
    "mat_nodistance":  (0.50, 0.00, 0.50),
    "mat_noattention": (0.00, 0.50, 0.50),
}


def make_model(d_atom: int, N: int = 2, d_model: int = 128, h: int = 8, dropout: float = 0.1,
               lambda_attention: float = 0.33, lambda_distance: float = 0.33, lambda_adjacency: float | None = None,
               N_dense: int = 2, leaky_relu_slope: float = 0.0, aggregation_type: str = "mean",
               dense_output_nonlinearity: str = "relu", distance_matrix_kernel: str = "softmax",
               n_output: int = 1, init_type: str = "uniform", n_generator_layers: int = 1) -> GraphTransformer:
    attn = MultiHeadedAttention(h, d_model, dropout, lambda_attention, lambda_distance, lambda_adjacency,
                                distance_matrix_kernel)
    ff = PositionwiseFeedForward(d_model, N_dense, dropout, leaky_relu_slope, dense_output_nonlinearity)
    model = GraphTransformer(
        Encoder(EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout), N),
        Embeddings(d_model, d_atom, dropout),
        Generator(d_model, aggregation_type, n_output, n_generator_layers, leaky_relu_slope, dropout),
    )
    for p in model.parameters():
        if p.dim() > 1:
            if init_type == "uniform":
                nn.init.xavier_uniform_(p)
            elif init_type == "normal":
                nn.init.xavier_normal_(p)
            else:
                raise ValueError(init_type)
    return model


def make_mat_variant(variant: str, d_atom: int = C.D_ATOM, d_model: int = 64, N: int = 2, h: int = 4,
                     dropout: float = 0.1, distance_matrix_kernel: str = "softmax", **kw) -> GraphTransformer:
    """Build MAT or one of its ablations with the lambda triplet from ABLATION_LAMBDAS."""
    la, ld, lg = ABLATION_LAMBDAS[variant]
    return make_model(d_atom=d_atom, N=N, d_model=d_model, h=h, dropout=dropout, lambda_attention=la,
                      lambda_distance=ld, lambda_adjacency=lg, N_dense=kw.pop("N_dense", 1),
                      leaky_relu_slope=kw.pop("leaky_relu_slope", 0.1),
                      distance_matrix_kernel=distance_matrix_kernel, **kw)


def make_pretrained_arch(d_atom: int = C.D_ATOM, n_output: int = 1, dropout: float | None = None) -> GraphTransformer:
    """The exact architecture of the released checkpoint (d_model 1024, 8 layers, 16 heads)."""
    arch = dict(C.PRETRAINED_ARCH)
    if dropout is not None:
        arch["dropout"] = dropout
    return make_model(d_atom=d_atom, n_output=n_output, **arch)


def load_pretrained(model: GraphTransformer, path=C.PRETRAINED_PATH, strict_encoder: bool = True) -> dict:
    """Copy every non-generator tensor of the released checkpoint into `model` (as the authors do)."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    own = model.state_dict()
    copied, skipped = [], []
    for name, param in state.items():
        if name.startswith("generator"):
            skipped.append(name)
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        assert name in own, f"pretrained key {name} missing in model"
        assert own[name].shape == param.shape, f"shape mismatch for {name}: {own[name].shape} vs {param.shape}"
        own[name].copy_(param)
        copied.append(name)
    model.load_state_dict(own)
    encoder_keys = [k for k in own if not k.startswith("generator")]
    missing = sorted(set(encoder_keys) - set(copied))
    if strict_encoder:
        assert not missing, f"encoder keys not covered by the checkpoint: {missing}"
    return {"n_copied": len(copied), "n_skipped_generator": len(skipped), "missing_encoder_keys": missing}
