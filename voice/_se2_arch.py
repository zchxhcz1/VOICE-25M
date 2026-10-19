"""VOICE SE(2) distance attention, as used by the 25M checkpoint."""
import math
import torch
from torch import nn
class GaussianModule(nn.Module):
    def __init__(self, d_pair: int, n_gaussians: int = 128, max_dist: float = 137.0):
        super().__init__()
        self.max_dist = max_dist
        self.means = nn.Parameter(torch.linspace(0.0, 1.0, n_gaussians))
        self.log_stds = nn.Parameter(torch.zeros(n_gaussians))
        self.proj = nn.Linear(n_gaussians, d_pair)

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        D_norm = (D / self.max_dist).clamp(0.0, 1.0)
        stds = self.log_stds.exp().clamp(min=1e-4)
        basis = torch.exp(-0.5 * ((D_norm.unsqueeze(-1) - self.means) / stds) ** 2)
        return self.proj(basis)

class SE2Layer(nn.Module):
    def __init__(self, d_model: int, d_pair: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.pair_update = nn.Linear(d_model, n_heads)
        self.pair_mlp = nn.Sequential(
            nn.Linear(d_pair + n_heads, d_pair), nn.GELU(), nn.Linear(d_pair, d_pair)
        )
        self.pair_to_bias = nn.Linear(d_pair, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_pair = nn.LayerNorm(d_pair)

    def forward(self, feat, Z):
        N = feat.shape[0]
        feat_n = self.norm1(feat)
        pair_update = self.pair_update(feat_n.unsqueeze(1) + feat_n.unsqueeze(0))
        Z = Z + self.pair_mlp(torch.cat([self.norm_pair(Z), pair_update], dim=-1))

        attn_bias = self.pair_to_bias(Z).permute(2, 0, 1)
        Q = self.q_proj(feat_n).view(N, self.n_heads, self.d_head).permute(1, 0, 2)
        K = self.k_proj(feat_n).view(N, self.n_heads, self.d_head).permute(1, 0, 2)
        V = self.v_proj(feat_n).view(N, self.n_heads, self.d_head).permute(1, 0, 2)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_head)
        scores = scores + attn_bias
        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(attn, V).permute(1, 0, 2).reshape(N, -1)
        feat = feat + self.out_proj(out)
        feat = feat + self.ffn(self.norm2(feat))
        return feat, Z
