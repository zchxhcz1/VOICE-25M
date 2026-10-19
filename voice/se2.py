"""Six spatial attention layers using pairwise distances in microns."""
import torch
from torch import nn
from voice._se2_arch import SE2Layer, GaussianModule


class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gaussian = GaussianModule(64, 128, 137.0)
        self.pre = nn.ModuleList([SE2Layer(256, 64, 8) for _ in range(3)])
        self.post = nn.ModuleList([SE2Layer(256, 64, 8) for _ in range(3)])
        self.norm_out = nn.LayerNorm(256)

    def forward(self, x, pos):
        Z = self.gaussian((pos.unsqueeze(0) - pos.unsqueeze(1)).norm(dim=-1))
        for layer in self.pre:
            x, Z = layer(x, Z)
        for layer in self.post:
            x, Z = layer(x, Z)
        return self.norm_out(x)
