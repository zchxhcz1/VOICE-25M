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
    def __init__(self, gene_emb, n_plat=2, d_in=1536, d_hid=1024, d_plat=32,
                 use_area=True, use_plat_in=True, use_plat_main=True, use_plat_gene=True,
                 learn_gamma=False, gamma_from_z=False, free_head=False, free_init="default"):
        super().__init__()
        G, D = gene_emb.shape
        self.register_buffer("E", torch.as_tensor(gene_emb, dtype=torch.float32))
        self.W = None
        if free_head:
            self.W = nn.Parameter(torch.empty(G, D))
            if free_init == "emb":
                with torch.no_grad(): self.W.copy_(self.E)
            else:
                nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.use_area, self.use_plat_in = use_area, use_plat_in
        self.use_plat_main, self.use_plat_gene = use_plat_main, use_plat_gene
        d = d_in + (d_plat if use_plat_in else 0) + (2 if use_area else 0)
        self.plat_emb = nn.Embedding(n_plat, d_plat) if use_plat_in else None
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_hid), nn.GELU(), nn.Linear(d_hid, D))
        self.b = nn.Parameter(torch.zeros(G))                       # gene main effect
        self.d = nn.Parameter(torch.zeros(n_plat)) if use_plat_main else None
        self.c = nn.Parameter(torch.zeros(n_plat, G)) if use_plat_gene else None
        self.log_gamma = nn.Parameter(torch.zeros(())) if (learn_gamma and not gamma_from_z) else None
        self.gamma_head = None
        self.gamma_lo, self.gamma_hi = 0.2, 2.0
        if gamma_from_z:
            self.gamma_head = nn.Linear(D, 1)
            nn.init.zeros_(self.gamma_head.weight)
            t = (1.0 - self.gamma_lo) / (self.gamma_hi - self.gamma_lo)       # sigmoid target for gamma = 1
            nn.init.constant_(self.gamma_head.bias, math.log(t / (1.0 - t)))
        self.log_theta = nn.Parameter(torch.zeros(G))               # per-gene NB dispersion

    def set_heldout(self, mask):
        """Exclude selected gene-specific parameters from training gradients."""
        self.register_buffer("_held", torch.as_tensor(mask, dtype=torch.bool), persistent=False)
        keep = (~torch.as_tensor(mask, dtype=torch.bool)).float()
        self.b.register_hook(lambda g, k=keep: g * k.to(g.device))
        if self.c is not None:
            self.c.register_hook(lambda g, k=keep: g * k.to(g.device)[None, :])

    def decay_params(self):
        """Only the interaction is decayed; b and d are main effects and must stay free."""
        return [self.c] if self.c is not None else []

    def forward(self, h, plat, area_abs, area_rel, head_idx, neutral=False):
        """Predict expected counts for the requested genes. neutral=True removes assay-specific output terms."""
        feats = [h]
        if self.use_plat_in:
            feats.append(self.plat_emb.weight.mean(0).expand(len(h), -1) if neutral
                         else self.plat_emb(plat))
        if self.use_area:
            feats.append(torch.stack([area_abs, area_rel], -1))
        z = self.mlp(torch.cat(feats, -1))                          # [B, D]
        E = (self.W if self.W is not None else self.E)[head_idx]    # [K, D]
        logit = z @ E.t() + self.b[head_idx]
        if self.d is not None and not neutral:
            logit = logit + self.d[plat][:, None]
        if self.c is not None and not neutral:
            logit = logit + self.c[plat][:, head_idx]
        mu = F.softplus(logit) + 1e-4
        if self.use_area:
            if self.gamma_head is not None:
                g = (self.gamma_lo + (self.gamma_hi - self.gamma_lo)
                     * torch.sigmoid(self.gamma_head(z))).squeeze(-1)    # [B] one bounded gamma per cell
                self._last_gamma = g.detach()
            elif self.log_gamma is not None:
                g = torch.exp(self.log_gamma)
            else:
                g = 1.0
            mu = mu * torch.exp(g * area_rel)[:, None]              # size factor: relative, not absolute
        return mu

def nb_nll(y, mu, log_theta):
    th = torch.exp(log_theta).clamp(1e-4, 1e4)
    return (torch.lgamma(th) + torch.lgamma(y + 1) - torch.lgamma(y + th)
            + th * torch.log1p(mu / th) + y * torch.log1p(th / mu))
