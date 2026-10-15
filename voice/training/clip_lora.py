"""Stage-1 pooling, projection and data utilities from the 25M run."""
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from voice.encoder import inject_lora, pooled_feat, MEAN, STD
TEMP=0.07
AMP_DTYPE=torch.bfloat16
dev=torch.device('cuda')
class HeTower(nn.Module):                 # Live UNI2-h feats -> LayerNorm handles scale (no precomputed he stats)
    def __init__(self, d=128): super().__init__(); self.ln = nn.LayerNorm(1536); self.net = nn.Sequential(nn.Linear(1536, 512), nn.GELU(), nn.Linear(512, d))
    def forward(self, x): return F.normalize(self.net(self.ln(x)), dim=-1)

class ScfTower(nn.Module):
    # Input width follows the target: 3072 for scFoundation (v25mfull), 512 for scGPT (clip_muon_ab --target scgpt).
    def __init__(self, mu, sd, d=128):
        super().__init__(); self.register_buffer("mu", mu); self.register_buffer("sd", sd)
        self.net = nn.Sequential(nn.Linear(mu.numel(), 512), nn.GELU(), nn.Linear(512, d))
    def forward(self, x): return F.normalize(self.net((x - self.mu) / self.sd), dim=-1)

class CLIPDS(Dataset):
    def __init__(self, paths, W, scf): self.paths = paths; self.W = W; self.scf = scf
    def __len__(self): return len(self.paths)
    def __getitem__(self, k):
        img = np.asarray(Image.open(self.paths[k]).convert("RGB").resize((224, 224), Image.BILINEAR), np.uint8)
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                torch.from_numpy(self.W[k]), torch.from_numpy(self.scf[k].astype(np.float32)))

def norm(x_uint8): x = x_uint8.float().to(dev, non_blocking=True) / 255.0; return (x - MEAN.to(dev)) / STD.to(dev)

def scf_stats(scf, chunk=200000):                                         # mean/std of a (possibly memmap) [N,3072] in chunks (low RAM)
    n, d = scf.shape; s = np.zeros(d, np.float64); ss = np.zeros(d, np.float64)
    for i in range(0, n, chunk):
        c = np.asarray(scf[i:i + chunk], np.float32); s += c.sum(0); ss += (c * c).sum(0)
    mu = s / n; var = np.maximum(ss / n - mu * mu, 0)
    return torch.from_numpy(mu.astype(np.float32)).to(dev), torch.from_numpy((np.sqrt(var) + 1e-6).astype(np.float32)).to(dev)

def infonce(zi, zg):
    logits = (zi @ zg.t()) / TEMP; lab = torch.arange(len(zi), device=zi.device)
    return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))

def he_embed(model, he_tower, img, W):    # crop -> LoRA UNI2-h -> cmask pool -> he tower [c,128]
    return he_tower(pooled_feat(model, norm(img), W.to(dev, non_blocking=True)))

def trainable(model): return {n: p for n, p in model.named_parameters() if p.requires_grad}

def load_slide_npz(sample_dir, per, seed):
    """Uniform-random `per` cells, cmask from the PRE-SAVED CELL polygons (patch_cell_boundaries.npz, patch coords).
    CELL (not nucleus) is what the cached frozen features used — verified cos=1.0 vs config_real features.
    Robust (no outs.zip derivation -> ovary etc. work) and ~250x faster than affine-rasterize. Returns (paths, W[.,16,16], er)."""
    z = np.load(os.path.join(sample_dir, "patch_cell_boundaries.npz"), allow_pickle=True)
    er_all = z["expr_rows"].astype(np.int64); ip = z["indptr"]; vx = z["vertex_x_patch"]; vy = z["vertex_y_patch"]
    osize = int(z["output_size"]); tok = osize // 16; n = len(er_all)
    sel = np.arange(n) if n <= per else np.sort(np.random.RandomState(seed).choice(n, per, replace=False))   # spatially-uniform (random over ALL cells)
    man = pd.read_csv(os.path.join(sample_dir, "manifest.csv.gz"), usecols=["expr_row", "patch_path"]).drop_duplicates("expr_row").set_index("expr_row")["patch_path"]
    paths, Ws, ers = [], [], []
    for i in sel:
        er = int(er_all[i]); a, b = int(ip[i]), int(ip[i + 1]); w = np.zeros((16, 16), np.float32)
        if b - a >= 3:
            im = Image.new("L", (osize, osize), 0); ImageDraw.Draw(im).polygon(list(zip(vx[a:b].tolist(), vy[a:b].tolist())), fill=1)
            w = np.asarray(im, np.float32).reshape(16, tok, 16, tok).mean((1, 3))
        if w.sum() <= 1e-6: w[8, 8] = 1.0                                  # degenerate -> central token
        Ws.append((w / w.sum()).astype(np.float32)); paths.append(os.path.join(sample_dir, str(man.loc[er]))); ers.append(er)
    return paths, np.stack(Ws), np.array(ers, np.int64)
