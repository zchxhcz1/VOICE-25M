"""Negative-binomial expression head."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
def cell_areas(d, er):
    """Shoelace area in the resampled patch frame, um^2, for the cells `er` indexes."""
    z = np.load(f"{d}/patch_cell_boundaries.npz", allow_pickle=True)
    ip = z["indptr"]; vx = z["vertex_x_patch"]; vy = z["vertex_y_patch"]
    er_all = z["expr_rows"].astype(np.int64)
    pos = {int(v): i for i, v in enumerate(er_all)}
    um2 = (55.0 / float(z["output_size"])) ** 2
    out = np.zeros(len(er), np.float32)
    for k, e in enumerate(er):
        i = pos[int(e)]; a, b = int(ip[i]), int(ip[i + 1])
        if b - a >= 3:
            x, y = vx[a:b], vy[a:b]
            out[k] = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) * um2
    med = np.median(out[out > 0]) if (out > 0).any() else 1.0
    out[out <= 0] = med                                   # degenerate polygon -> slide median, not 0
    return out, float(med)

class DirectHead(nn.Module):
    """Learnable scGPT gene rows, cell area covariates and a global area exponent."""
    def __init__(self, gene_emb):
        super().__init__()
        G, D = gene_emb.shape
        self.register_buffer('E', torch.as_tensor(gene_emb, dtype=torch.float32))
        self.W = nn.Parameter(self.E.clone())
        self.mlp = nn.Sequential(nn.LayerNorm(258), nn.Linear(258, 1024), nn.GELU(), nn.Linear(1024, D))
        self.b = nn.Parameter(torch.zeros(G))
        self.log_gamma = nn.Parameter(torch.zeros(()))
        self.log_theta = nn.Parameter(torch.zeros(G))

    def forward(self, h, area_abs, area_rel, head_idx):
        z = self.mlp(torch.cat([h, torch.stack([area_abs, area_rel], -1)], -1))
        logit = z @ self.W[head_idx].t() + self.b[head_idx]
        mu = F.softplus(logit) + 1e-4
        return mu * torch.exp(torch.exp(self.log_gamma) * area_rel)[:, None]

def nb_nll(y, mu, log_theta):
    th = torch.exp(log_theta).clamp(1e-4, 1e4)
    return (torch.lgamma(th) + torch.lgamma(y + 1) - torch.lgamma(y + th)
            + th * torch.log1p(mu / th) + y * torch.log1p(th / mu))
