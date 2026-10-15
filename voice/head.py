"""25M negative-binomial expression head."""
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
        # free_head: the gene rows become a learnable [G, D] matrix W, so logit[g] = <z, w_g> + b[g].
        # free_init="emb" starts W from the table passed as gene_emb (scGPT gene embeddings for the abl5m winner,
        # docs/scgpt_variant.md); "default" uses nn.Linear's init. Without free_head the table stays frozen,
        # which is how live_nod4_c was trained (scFoundation pos_emb).
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
        # size-factor exponent. mu = softplus(logit) * exp(gamma * area_rel); gamma = 1 hard-codes
        # "counts scale linearly with area", but the measured within-slide slope of log(total count) on
        # log(area) runs 0.47 (kidney) to 1.35 (HCC) over 13 objects, mean 0.93 -- so a fixed 1.0
        # over-corrects the shallow slides by nearly 2x. One learnable scalar fixes the average.
        #
        # It is deliberately GLOBAL, not per-slide: a per-slide gamma has no value to look up on a new
        # slide, the same reason the per-slide count median was rejected as a size factor. The residual
        # slide-to-slide slope variation is not learnable in a deployable form; a gamma predicted from the
        # image would be (tissue is visible), but that is a bigger change than this ablation needs.
        # Global gamma is one free scalar: its gradient is sum(residual * area_rel), so it settles where the
        # residual is uncorrelated with relative cell size. That removes the AVERAGE dependence and nothing
        # more, and the measured within-slide slope spans 0.47 (kidney) to 1.35 (ovary Prime) -- 2.8x -- so a
        # single value is a compromise that is wrong on both ends.
        #
        # gamma_from_z predicts it per cell from the image features instead. Deployable for the same reason
        # the global one is: no metadata, no per-slide value to look up. The bias is initialised so that
        # softplus(bias) = 1.0 and the weight at zero, so every cell starts at gamma = 1.0 exactly like the
        # global arm -- the two differ only in whether gamma is allowed to vary.
        self.log_gamma = nn.Parameter(torch.zeros(())) if (learn_gamma and not gamma_from_z) else None
        # BOUNDED, because the first attempt at this diverged. With gamma = softplus(w.z + b) there is no
        # upper limit, and the size factor is exp(gamma * area_rel) with area_rel spanning about +-3: gamma
        # only has to reach ~40 for exp() to overflow fp32. It did -- gamma went 1.055 -> 2.108 -> nan
        # between steps 500 and 1500 and every metric collapsed to 0.
        #
        # A single global gamma cannot do this: 5.3M cells constrain one scalar, and it settled at 0.753.
        # A per-cell gamma is barely constrained for cells whose area_rel is near zero -- the gradient in
        # that direction vanishes -- so it is free to drift until it overflows.
        #
        # sigmoid bounds it to [lo, hi] instead. The measured within-slide slopes run 0.473-1.310, so
        # [0.2, 2.0] is roomy: this constrains the failure mode, not the hypothesis. The bias is set so
        # gamma starts at exactly 1.0, matching the global arm's initialisation.
        self.gamma_head = None
        self.gamma_lo, self.gamma_hi = 0.2, 2.0
        if gamma_from_z:
            self.gamma_head = nn.Linear(D, 1)
            nn.init.zeros_(self.gamma_head.weight)
            t = (1.0 - self.gamma_lo) / (self.gamma_hi - self.gamma_lo)       # sigmoid target for gamma = 1
            nn.init.constant_(self.gamma_head.bias, math.log(t / (1.0 - t)))
        self.log_theta = nn.Parameter(torch.zeros(G))               # per-gene NB dispersion

    def set_heldout(self, mask):
        """mask[G] bool: genes whose OWN parameters must never be trained.

        The head is logit[g] = <z, e_g> + b[g] + d[p] + c[p,g]. Everything gene-specific lives in b[g] and
        c[p,g]; e_g is a frozen scFoundation row and z is shared. Zeroing the gradient of b[g] and c[p,g]
        for a held-out set leaves those genes predictable ONLY through <z, e_g> -- which is the actual claim
        behind "genes a slide's panel does not measure are still predictable". If held-out genes score near
        the trained ones, the embedding head is doing the work; if they collapse, the cross-panel ability
        was really coming from a free parameter per gene, and the claim needs restating.
        """
        # set_heldout is called AFTER .to(device), so a tensor built here stays on the CPU while the
        # gradient arrives on the GPU. The mask is moved onto the gradient's own device inside the hook.
        self.register_buffer("_held", torch.as_tensor(mask, dtype=torch.bool), persistent=False)
        keep = (~torch.as_tensor(mask, dtype=torch.bool)).float()
        self.b.register_hook(lambda g, k=keep: g * k.to(g.device))
        if self.c is not None:
            self.c.register_hook(lambda g, k=keep: g * k.to(g.device)[None, :])

    def decay_params(self):
        """Only the interaction is decayed; b and d are main effects and must stay free."""
        return [self.c] if self.c is not None else []

    def forward(self, h, plat, area_abs, area_rel, head_idx, neutral=False):
        """neutral=True drops every platform-specific term, which is what predicting on a slide from NEITHER
        platform requires -- a plain H&E section has no assay identity to look up.

        The output-side terms are removable by construction: d[p] and c[p,g] are additive inside the softplus,
        so zeroing them leaves b[g] carrying the platform-free gene level. plat_in is not: it feeds the MLP,
        so substituting a zero vector is an out-of-distribution input rather than a neutral one. The mean
        embedding is used instead, which is the least-bad choice and still not a principled neutral point.
        That asymmetry is the reason plat_in and (d, c) are not interchangeable in the ablation: an arm can
        win on the validation slides and still be the worse model for the deployment case."""
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
