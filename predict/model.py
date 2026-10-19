"""The released 25M architecture, without importing training drivers."""
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from voice.encoder import build_uni2_local, inject_lora, pooled_feat, MEAN, STD
from voice.head import DirectHead
from voice.se2 import SpatialTransformer
from voice.recipe import check_stage2_settings
from voice.training_contract import load_lora_checked, gene_hash


class Voice25M:
    def __init__(self, release, device='cuda', amp='bf16'):
        self.device = torch.device(device)
        self.amp = amp
        self.space = np.load(release.paths['gene_space'], allow_pickle=False)
        self.genes = self.space['genes'].astype(str)
        if len(set(self.genes)) != len(self.genes):
            raise ValueError('Duplicate genes in gene space')
        ck = torch.load(release.paths['stage2'], map_location='cpu', weights_only=False)
        s1 = torch.load(release.paths['stage1'], map_location='cpu', weights_only=False)
        expected = ck['stage1_identity']['sha256']
        if expected != release.hashes['stage1']:
            raise ValueError('Stage-2 was trained with another Stage-1 initializer')
        if ck['gene_head_sha256'] != gene_hash(self.genes.tolist()):
            raise ValueError('Stage-2 gene order mismatch')
        self.args = a = ck['args']
        check_stage2_settings(a)
        enc = build_uni2_local(self.device, str(release.paths['uni2h']))
        inject_lora(enc, a['nblocks'], a['lora_r'], a['lora_alpha'], a['lora_dropout'])
        enc.to(self.device)
        load_lora_checked(enc, s1['lora'], complete=True)
        load_lora_checked(enc, ck['lora'])
        self.enc = enc.eval()
        self.inp = nn.Linear(1536, a['d_model']).to(self.device)
        self.se2 = SpatialTransformer().to(self.device)
        self.head = DirectHead(np.zeros(tuple(ck['head']['E'].shape), np.float32)).to(self.device)
        for key in ['inp', 'se2', 'head']:
            getattr(self, key).load_state_dict(ck[key], strict=True)
            getattr(self, key).eval()
        params = dict(enc.named_parameters())
        self._params = {k: params[k] for k in s1['lora']}
        self._p1 = {k: v.to(self.device, params[k].dtype) for k, v in s1['lora'].items()}
        # Stage-2 saves only the trainable subset; frozen adapters still come from Stage-1.
        self._s2 = {k: v.to(self.device, params[k].dtype) for k, v in ck['lora'].items()}
        self.step = int(ck['step'])
        self._which = None

    @torch.no_grad()
    def encode(self, images, masks, which='stage2', chunk=64):
        if which != self._which:
            for k, v in (self._p1 if which == 'phase1' else self._s2).items():
                self._params[k].copy_(v)
            self._which = which
        ctx = (torch.autocast('cuda', dtype=torch.bfloat16 if self.amp == 'bf16' else torch.float16)
               if self.device.type == 'cuda' and self.amp != 'fp32' else nullcontext())
        mean, std = MEAN.to(self.device), STD.to(self.device)
        with ctx:
            features = [pooled_feat(self.enc, (images[i:i+chunk].float()/255 - mean)/std,
                                   masks[i:i+chunk]) for i in range(0, len(images), chunk)]
        return torch.cat(features).float()

    @torch.no_grad()
    def decode(self, features, positions, area_abs, area_rel, gene_ids):
        z = self.se2(self.inp(features), positions)
        mu = self.head(z, area_abs, area_rel, gene_ids)
        return torch.log1p(mu), z
