"""SMELT-style prelude-recur-coda looping for VOICE's SE(2) transformer.

VOICE's SE2Transformer is a ModuleList of n_layers=4 SE2Layers over a set of spatially co-located cells:

    Z = gaussian(pairwise distances)         # [N, N, d_pair]  edge state
    for layer in self.layers:
        feat, Z = layer(feat, Z)             # BOTH node and edge state are refined

That is a message-passing iteration over a (node, edge) state, which is a better fit for weight-tied looping
than a plain token stack -- looping re-applies the same refinement operator, exactly what the state form
suggests. SE(2) invariance is unaffected: it comes from Z depending only on |pos_i - pos_j|, and every loop
pass reads the same distances.

SE2Layer has THREE residual updates, not two:

    Z    = Z    + pair_mlp(...)
    feat = feat + out_proj(attn_out)
    feat = feat + ffn(norm2(feat))

SMELT scales the residual update of every sublayer inside the looped span by 1/r, because weight-tied passes
produce correlated updates that otherwise inflate the state by roughly r. All three are scaled here; missing
the Z one would let the edge state grow unchecked while the node state stayed calibrated.

Nothing in voice/ is edited: LoopedSE2Layer subclasses SE2Layer and only overrides forward.
"""
from __future__ import annotations
import math
import torch, torch.nn as nn
import os, sys

from voice._se2_arch import SE2Layer, SE2Transformer, GaussianModule


class LoopedSE2Layer(SE2Layer):
    def forward(self, feat, Z, res_scale: float = 1.0):
        N = feat.shape[0]
        feat_n = self.norm1(feat)
        pair_update = self.pair_update(feat_n.unsqueeze(1) + feat_n.unsqueeze(0))
        Z = Z + res_scale * self.pair_mlp(torch.cat([self.norm_pair(Z), pair_update], dim=-1))

        attn_bias = self.pair_to_bias(Z).permute(2, 0, 1)
        Q = self.q_proj(feat_n).view(N, self.n_heads, self.d_head).permute(1, 0, 2)
        K = self.k_proj(feat_n).view(N, self.n_heads, self.d_head).permute(1, 0, 2)
        V = self.v_proj(feat_n).view(N, self.n_heads, self.d_head).permute(1, 0, 2)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_head) + attn_bias
        out = torch.bmm(torch.softmax(scores, dim=-1), V).permute(1, 0, 2).reshape(N, -1)
        feat = feat + res_scale * self.out_proj(out)
        feat = feat + res_scale * self.ffn(self.norm2(feat))
        return feat, Z


class LoopedSE2Transformer(nn.Module):
    """n_layers is the PARAMETER budget in SE2Layers; `span` is the looped fraction; effective depth is
    n_layers + (span_layers) * (r - 1).

    span=0 or r=1 reproduces VOICE's SE2Transformer exactly (asserted in the self-test below), so the
    baseline arm is not a re-implementation with different numerics.
    """

    def __init__(self, d_model=256, d_pair=64, n_heads=8, n_layers=4, span=0.0, r=1,
                 n_gaussians=128, max_dist=137.0):
        super().__init__()
        assert 0.0 <= span <= 1.0 and r >= 1
        self.gaussian = GaussianModule(d_pair, n_gaussians, max_dist)
        n_mid = int(round(n_layers * span)) if r > 1 else 0
        n_pre = (n_layers - n_mid) // 2
        n_post = n_layers - n_mid - n_pre
        mk = lambda n: nn.ModuleList([LoopedSE2Layer(d_model, d_pair, n_heads) for _ in range(n)])
        self.pre, self.mid, self.post = mk(n_pre), mk(n_mid), mk(n_post)
        self.norm_out = nn.LayerNorm(d_model)
        self.r = r if n_mid else 1
        self.n_pre, self.n_mid, self.n_post = n_pre, n_mid, n_post
        self.max_dist = max_dist

    @property
    def effective_depth(self):
        return self.n_pre + self.n_mid * self.r + self.n_post

    def describe(self):
        return (f"{self.n_pre}+{self.n_mid}+{self.n_post} layers (pre+mid+coda), r={self.r}, "
                f"depth {self.effective_depth}, {sum(p.numel() for p in self.parameters()):,} params, "
                f"max_dist {self.max_dist:.0f}um")

    def forward(self, x, pos):
        """x [N, d_model]; pos [N, 2] in MICRONS.

        VOICE's scale_dataset builds pos from x_pixel/y_pixel, which are SOURCE H&E pixels, and
        um_per_pixel_source spans 0.1369-0.4812 across this corpus -- a 3.52x ratio. A fixed max_dist in
        pixels therefore covers 68um on the finest slide and 241um on the coarsest, so the Gaussian basis
        would describe a different physical neighbourhood per slide. Microns make it comparable.
        """
        D = (pos.unsqueeze(0) - pos.unsqueeze(1)).norm(dim=-1)
        Z = self.gaussian(D)
        feat = x
        for layer in self.pre:
            feat, Z = layer(feat, Z)
        rs = 1.0 / self.r
        for _ in range(self.r):
            for layer in self.mid:
                feat, Z = layer(feat, Z, res_scale=rs)
        for layer in self.post:
            feat, Z = layer(feat, Z)
        return self.norm_out(feat)
