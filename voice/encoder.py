"""UNI2-h and LoRA utilities shared by training and inference."""
import os
import torch
from torch import nn
import timm
MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
STD = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
PREFIX = 9
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout=0.05):
        super().__init__(); self.base = base
        for p in base.parameters(): p.requires_grad = False
        self.A = nn.Parameter(torch.zeros(r, base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5); self.s = alpha / r; self.drop = nn.Dropout(dropout)
    def forward(self, x): return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.s

def inject_lora(m, nblocks, r, alpha, dropout):
    n = len(m.blocks)
    for i in range(n - nblocks, n):
        m.blocks[i].attn.qkv = LoRALinear(m.blocks[i].attn.qkv, r, alpha, dropout)
        m.blocks[i].attn.proj = LoRALinear(m.blocks[i].attn.proj, r, alpha, dropout)
def pooled_feat(model, x, w16):
    """x:[B,3,224,224] normed; w16:[B,16,16] mask weights (sum=1). Returns [B,1536] (recipe cmask pool)."""
    t = model.forward_features(x)                     # [B, 265, 1536]
    patch = t[:, PREFIX:]                             # [B, 256, 1536]
    side = int(round(patch.shape[1] ** 0.5))
    grid = patch.reshape(patch.shape[0], side, side, -1)
    return (grid * w16.unsqueeze(-1)).sum((1, 2))
def build_uni2_local(dev, path):
    """Build UNI2-h and load an explicit local state dict with strict shapes."""
    m = timm.create_model("vit_huge_patch14_224", pretrained=False, img_size=224, patch_size=14, depth=24,
                          num_heads=24, init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0,
                          no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
                          reg_tokens=8, dynamic_img_size=True)
    sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    r = m.load_state_dict(sd, strict=True)
    for p_ in m.parameters(): p_.requires_grad = False
    print(f"[uni2h] loaded {path} ({sum(p_.numel() for p_ in m.parameters()):,} params, strict)", flush=True)
    return m.to(dev).eval()
