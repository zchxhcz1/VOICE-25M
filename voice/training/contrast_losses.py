"""Contrastive training losses."""
import math
import torch, torch.nn as nn, torch.nn.functional as F
import clip_lora as C


class ContrastLoss(nn.Module):
    def __init__(self, kind: str, temp: float = C.TEMP):
        super().__init__()
        self.kind = kind
        self.temp = temp
        if kind == "learntemp":
            # CLIP's parameterisation: learn log(1/T) and exponentiate, so the scale stays positive without a
            # constraint. Init to the fixed value the other arms use, so this arm STARTS as the baseline and
            # any difference is attributable to the temperature having moved.
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temp)))
        elif kind == "siglip":
            # SigLIP's published init: t' = log 10, b = -10. The bias matters -- at init almost every pair is
            # a negative (n-1 of n per row), and without a large negative bias the loss is dominated by them.
            self.t_prime = nn.Parameter(torch.tensor(2.3025851))     # log 10
            self.bias = nn.Parameter(torch.tensor(-10.0))
        elif kind not in ("infonce", "dcl"):
            raise ValueError(kind)

    def extra_repr(self): return f"kind={self.kind}"

    def forward(self, zi, zg):
        n = zi.shape[0]
        lab = torch.arange(n, device=zi.device)
        if self.kind == "infonce":
            return C.infonce(zi, zg)                       # exactly the baseline, not a re-implementation
        if self.kind == "learntemp":
            logits = (zi @ zg.t()) * self.logit_scale.exp().clamp(max=100.0)
            return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
        if self.kind == "dcl":
            # Decoupled contrastive learning: drop the positive from its own denominator. In softmax InfoNCE
            # the positive term appears in the normaliser, so a row whose positive already dominates yields a
            # small gradient exactly where the negatives still need pushing apart. Removing it keeps the
            # negative-repulsion gradient alive; the effect is largest at small negative pools, which is the
            # regime here (256 in-batch candidates).
            logits = (zi @ zg.t()) / self.temp
            eye = torch.eye(n, dtype=torch.bool, device=zi.device)
            def one_way(lg):
                pos = lg.diagonal()
                neg = lg.masked_fill(eye, float("-inf"))
                return (-pos + torch.logsumexp(neg, dim=1)).mean()
            return 0.5 * (one_way(logits) + one_way(logits.t()))
        # siglip: every pair scored independently with a sigmoid, so there is no softmax over a negative pool
        # and no dependence on batch composition. Normalised by n (not n^2) as in the paper.
        logits = (zi @ zg.t()) * self.t_prime.exp() + self.bias
        labels = 2.0 * torch.eye(n, device=zi.device) - 1.0
        return -F.logsigmoid(labels * logits).sum() / n
