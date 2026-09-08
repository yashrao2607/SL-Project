"""ECFP + MAT late-fusion hybrid.

The MAT encoder produces a masked-mean pooled molecular embedding; an ECFP4 fingerprint is projected
to a small dense vector; the two representations are concatenated ("late fusion" at the
representation level) and fed to a single linear output layer, exactly like MAT's own generator, so the
comparison against MAT isolates the effect of the fingerprint input. Everything is trained end-to-end.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .mat import GraphTransformer, masked_mean_pool


class ECFPMATHybrid(nn.Module):
    def __init__(self, mat: GraphTransformer, d_model: int, n_bits: int = 2048, d_fp: int = 128,
                 dropout: float = 0.1, n_output: int = 1, leaky_relu_slope: float = 0.1):
        super().__init__()
        self.mat = mat
        self.mat.generator = nn.Identity()          # MAT's own output head is not used (avoids dead parameters)
        self.fp_proj = nn.Sequential(nn.Linear(n_bits, d_fp), nn.ReLU(), nn.Dropout(dropout))
        # Linear head on the concatenation, mirroring MAT's single linear generator, so that the only
        # difference between the hybrid and plain MAT is the fingerprint input (late fusion).
        self.head = nn.Linear(d_model + d_fp, n_output)

    def forward(self, src, src_mask, adj_matrix, distances_matrix, fp):
        h = self.mat.encode(src, src_mask, adj_matrix, distances_matrix)
        g = masked_mean_pool(h, src_mask)
        return self.head(torch.cat([g, self.fp_proj(fp)], dim=-1))
