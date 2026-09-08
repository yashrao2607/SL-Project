"""Graph Convolutional Network baseline (Kipf & Welling) built with PyTorch Geometric."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, n_layers: int = 3, dropout: float = 0.1, n_output: int = 1):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(n_layers)])
        self.dropout = dropout
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_output))

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        return self.head(g)

    def embed(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)


def record_to_pyg(rec: dict, y: float) -> Data:
    """Convert a cached MAT feature record (with dummy node) into a PyG graph without the dummy node."""
    afm = rec["afm"][1:, 1:]                       # drop dummy row and indicator column
    adj = rec["adj"][1:, 1:].copy()
    np.fill_diagonal(adj, 0.0)                     # GCNConv adds its own self loops
    src, dst = np.nonzero(adj)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long) if len(src) else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=torch.tensor(afm, dtype=torch.float32), edge_index=edge_index,
                y=torch.tensor([y], dtype=torch.float32))
