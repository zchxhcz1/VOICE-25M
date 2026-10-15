"""Stage-1 scGPT alignment with AdamW, D4 and complete cell coverage."""
from __future__ import annotations
import os, sys, json, time, argparse, types, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, torch, torch.nn.functional as F
import scipy.sparse as sp
import torch.distributed as dist
from datetime import timedelta
from torch.utils.data import DataLoader, DistributedSampler

import clip_lora as C

from contrast_losses import ContrastLoss
import timm

DATASET = os.environ.get("VOICE_DATASET", "data")
SCF_EMB = os.environ.get("VOICE_SCF_EMB", "cell_embeddings")
SCF_ROOTS = [
    f"{SCF_EMB}/cosmx",         # regenerated CosMx (17 objects)
    f"{SCF_EMB}/cell_emb_scf",  # Xenium, transferred (84)
    f"{SCF_EMB}/xenium",
    f"{SCF_EMB}/cosmx",
]
SCF_ROOT = SCF_EMB  # legacy, kept for callers
CKDIR = os.environ.get("VOICE_CKDIR", "runs/clip_ab/ckpts")
UNI2H = os.environ.get("VOICE_UNI2H", "weights/uni2h")


def build_uni2_local(dev, path=UNI2H):
    from voice.encoder import build_uni2_local as build
    return build(dev, path)


# ------------------------------------------------------------------ data
def scf_path(slide, n_expr=None):
    """First embedding for `slide` across SCF_ROOTS; with n_expr given, the first whose row count matches.

    Passing n_expr matters where a slide has files in more than one root and only one is current -- the six
    CosMx TID objects are exactly that case.
    """
    for root in SCF_ROOTS:
        p = f"{root}/{slide}/scf_cellemb.npy"
        if os.path.exists(p) and (n_expr is None or np.load(p, mmap_mode="r").shape[0] == n_expr):
            return p
    return None


# Stage-1 alignment target. scf: scFoundation cell embeddings (3072-d, v25mfull). scgpt: scGPT whole-human <cls>
# embeddings (512-d) from scgpt_cellemb.py, ROW-ALIGNED with expression.npz like scf_cellemb.npy; the abl5m
# ablation chose this target (docs/scgpt_variant.md). Set by main() from --target.
SCGPT_EMB = os.environ.get("VOICE_SCGPT_EMB", "")
TARGET = "scf"


def target_path(slide, n_expr=None):
    """The current target's embedding file for `slide`, row-count checked like scf_path."""
    if TARGET == "scf":
        return scf_path(slide, n_expr)
    p = f"{SCGPT_EMB}/{slide}.npy"
    if os.path.exists(p) and (n_expr is None or np.load(p, mmap_mode="r").shape[0] == n_expr):
        return p
    return None


def usable_slides(split, platform="xenium"):
    """Objects in `split` whose scF embedding is ROW-ALIGNED with the object's expression matrix.

    6 CosMx objects (CosMx_HE_alignment/TID*) have MORE scF rows than cells: they were re-filtered during
    the dataset reassembly and their embeddings are stale. Row-misaligned scF would silently pair a crop
    with another cell's expression, so they are excluded here rather than trusted.
    """
    import glob, scipy.sparse as sp
    out = []
    for d in sorted(glob.glob(f"{DATASET}/{split}/{platform}/*/*/")):
        slide = os.path.basename(d.rstrip("/"))
        if not os.path.exists(d + "patch_cell_boundaries.npz"):
            continue
        n_expr = sp.load_npz(d + "expression.npz").shape[0]
        if target_path(slide, n_expr) is not None:
            out.append((slide, d.rstrip("/")))
    return out


def build_data(slides, max_cells, seed):
    """Same construction as clip_lora.build_data, but enumerating the new tree instead of SourceCache."""
    per = max(1, max_cells // max(len(slides), 1)) if max_cells else 2**62
    cache=os.environ.get('VOICE_TRAIN_CACHE')
    if cache:
        from voice.training_cache import cached_arrays, file_stamp, JoinedArrays
        arrays=[]
        for slide,d in slides:
            target=target_path(slide)
            signature={'target':file_stamp(target),'boundaries':file_stamp(f'{d}/patch_cell_boundaries.npz'),
                       'manifest':file_stamp(f'{d}/manifest.csv.gz'),'per':per,'seed':seed,'schema':1}
            def build(root):
                p,w,er=C.load_slide_npz(d,per,seed)
                sc=np.load(target,mmap_mode='r')[er].astype(np.float16)
                keep=np.where(np.abs(sc).sum(1)>0)[0]
                for name,value in {'paths':np.asarray([p[i] for i in keep],dtype=np.bytes_),
                                   'W':w[keep],'targets':sc[keep]}.items():
                    np.save(root/f'{name}.tmp.npy',value)
                return ['paths','W','targets']
            a=cached_arrays(f'{cache}/stage1/{slide}',signature,build);arrays.append(a)
            print(f'  [{slide[:38]:38s}] +{len(a["paths"]):>7d} shared mmap cells',flush=True)
        return tuple(JoinedArrays([a[key] for a in arrays]) for key in ['paths','W','targets'])
    W_l, scf_l, paths = [], [], []
    t0 = time.time()
    for slide, d in slides:
        p, w, er = C.load_slide_npz(d, per, seed)
        sc = np.load(target_path(slide, sp.load_npz(d + "/expression.npz").shape[0]), mmap_mode="r")[er].astype(np.float16)
        keep = np.where(np.abs(sc).sum(1) > 0)[0]                      # drop degenerate (all-zero) scF cells
        W_l.append(w[keep]); scf_l.append(sc[keep]); paths += [p[i] for i in keep]
        print(f"  [{slide[:38]:38s}] +{len(keep):>7d} ({time.time()-t0:.0f}s)", flush=True)
    return paths, np.concatenate(W_l), np.concatenate(scf_l)


# ------------------------------------------------------------------ D4 augmentation
class D4Wrap(torch.utils.data.Dataset):
    """A random element of D4 (4 rotations x 2 flips) applied to (crop, mask) TOGETHER.

    maskW is a 16x16 grid of weights over the ViT's 16x16 patch tokens, so it is a spatial object: turning the
    224x224 crop while leaving the mask fixed would pool the neighbouring cells' patches instead of this cell's,
    and the run would measure that bug rather than augmentation. 224 = 16 x 14, so a 90 deg image rotation maps
    patch (i,j) -> (j, 15-i), exactly what rot90 does to the mask -- the two stay aligned with no interpolation.

    Nothing is materialised: the corpus stays at its original size and each epoch still visits every cell once.
    What changes is that a cell appears in a different orientation on a later epoch, which is a regulariser, not
    more data -- it does not substitute for raising --max_cells.

    `fixed` assigns element (index mod 8) instead of drawing one, for a val pass that must be reproducible
    across evals and across arms.
    """

    def __init__(self, base, fixed=False, cycle=False, seed=0):
        self.base = base
        self.fixed = fixed
        self.cycle = cycle
        self.epoch = 0
        if cycle:
            # Sampling the 8 elements with replacement collides: after n exposures a cell has seen only
            # 8*(1-(7/8)^n) distinct ones -- 3.96 at 5.12 epochs, and 21.7 exposures are needed to expect
            # all 8. Cycling a per-cell permutation gives exactly min(n,8) instead, so 8 epochs covers the
            # orbit exactly once. It has to be a stateless function of (index, epoch): with num_workers>0
            # each worker forks its own copy of this object, so a counter would count independently in each.
            rng = np.random.default_rng(seed)
            self.perm = np.stack([rng.permutation(8) for _ in range(len(base))]).astype(np.uint8)

    def set_epoch(self, e):
        self.epoch = int(e)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, k):
        img, W, scf = self.base[k]
        if self.fixed:
            g = k % 8
            rot, flip = g % 4, g >= 4
        elif self.cycle:
            r = int(self.perm[k, self.epoch % 8])
            rot, flip = r % 4, r >= 4
        else:
            r = torch.randint(0, 8, (1,)).item()          # per-worker seeded by the DataLoader
            rot, flip = r % 4, r >= 4
        if rot:
            img = torch.rot90(img, rot, dims=(1, 2)); W = torch.rot90(W, rot, dims=(0, 1))
        if flip:
            img = img.flip(2); W = W.flip(1)
        return img.contiguous(), W.contiguous(), scf


# ------------------------------------------------------------------ optimizers
class MultiOpt:
    """Two optimizers driven as one. gradcache_step only needs zero_grad/step."""

    def __init__(self, opts): self.opts = [o for o in opts if o is not None]
    def zero_grad(self, set_to_none=True):
        for o in self.opts: o.zero_grad(set_to_none=set_to_none)
    def step(self):
        for o in self.opts: o.step()
    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]
    def state_dict(self): return [o.state_dict() for o in self.opts]
    def load_state_dict(self, states):
        if len(states) != len(self.opts):
            raise ValueError('Resume optimizer count differs')
        for optimizer, state in zip(self.opts, states):
            optimizer.load_state_dict(state)

    def set_lr_scale(self, scale):
        """Warmup. Each group's base lr is stamped on at construction, so this cannot depend on group order."""
        for g in self.param_groups:
            g["lr"] = g["base_lr"] * scale






def build_optimizer(kind, model, he_tower, scf_tower, lr_lora, lr_tower, wd, adjust_on="param", loss_params=()):
    if kind != 'adamw':
        raise ValueError('The released 25M recipe uses AdamW')
    lora = [p for p in model.parameters() if p.requires_grad]
    tower = list(he_tower.parameters()) + list(scf_tower.parameters()) + list(loss_params)
    opt = torch.optim.AdamW([{'params':lora,'lr':lr_lora}, {'params':tower,'lr':lr_tower}], weight_decay=wd)
    for group in opt.param_groups:
        group['base_lr'] = group['lr']
    return MultiOpt([opt])


# ------------------------------------------------------------------ distributed
def gradcache_step_ddp(chunks, model, he_tower, scf_tower, opt, world, lossfn, clip_norm=1.0):
    """clip_lora.gradcache_step with the all-reduce VOICE's own train_phase1.py uses.

    Standard DistributedDataParallel is wrong here and the repo says so: GradCache issues one backward PER
    CHUNK, and DDP's autograd hooks would all-reduce on every one of them -- wrong semantics, since a chunk's
    gradient is only a partial sum, and eight times the traffic. The gradients are instead reduced ONCE, after
    the last chunk, exactly as train_phase1.py:270-273.

    eff_batch stays PER RANK, so the InfoNCE negative pool and hence the task difficulty are unchanged from a
    single-card run; what doubles is the number of gradient samples behind each optimiser step.
    """
    with torch.no_grad(), torch.autocast("cuda", dtype=C.AMP_DTYPE):
        Zi = [C.he_embed(model, he_tower, b[0], b[1]).float() for b in chunks]
        Zg = [scf_tower(b[2].to(C.dev, non_blocking=True)).float() for b in chunks]
    zi = torch.cat(Zi).detach().requires_grad_(True)
    zg = torch.cat(Zg).detach().requires_grad_(True)
    # zero_grad must come BEFORE this backward: SigLIP and the learnable-temperature arm carry loss
    # parameters, they receive their ONLY gradient here in pass 2 (pass 3 re-forwards the towers, not the
    # loss), and zeroing afterwards would silently discard it -- the temperature would never move and the
    # arm would be a slow re-run of the baseline.
    opt.zero_grad(set_to_none=True)
    if len(chunks[0])==4:
        mask=torch.cat([b[3] for b in chunks]).to(C.dev)
        n=mask.sum();total=n.clone()
        if world>1:dist.all_reduce(total)
        loss=(lossfn(zi[mask],zg[mask]) if n.item() else (zi.sum()+zg.sum())*0)
        loss=loss*(world*n/total.clamp_min(1))
    else:
        loss=lossfn(zi,zg)
    loss.backward(); gi, gg = zi.grad, zg.grad
    off = 0
    for batch in chunks:
        img,W,scf=batch[:3]
        c = img.shape[0]
        with torch.autocast("cuda", dtype=C.AMP_DTYPE):
            zi_c = C.he_embed(model, he_tower, img, W)
            zg_c = scf_tower(scf.to(C.dev, non_blocking=True))
        torch.autograd.backward([zi_c.float(), zg_c.float()], [gi[off:off + c], gg[off:off + c]])
        off += c
    params = ([p for p in model.parameters() if p.requires_grad]
              + list(he_tower.parameters()) + list(scf_tower.parameters())
              + [q for q in lossfn.parameters() if q.requires_grad])
    if world > 1:                                          # ONE all-reduce, after the last chunk
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM); p.grad /= world
    torch.nn.utils.clip_grad_norm_(params, clip_norm); opt.step()
    return float(loss.detach())


# ------------------------------------------------------------------ eval
@torch.no_grad()
def evaluate(model, he_tower, scf_tower, loader, max_batches=200, world=1):
    """Val InfoNCE + in-batch R@1 both directions. Fixed batches (drop_last) so it is comparable across steps.

    max_batches was 8, i.e. 2048 cells, whose binomial standard error at R@1~0.11 is sqrt(.11*.89/2048)=0.0069
    -- the same size as the run-to-run differences being measured, and the same size as the observed jitter
    between adjacent evals, whose sd matched that figure almost exactly. Every argmax over such a curve was
    picking a sampling spike. Using the whole val set instead takes the error to ~0.0018.
    """
    model.eval(); he_tower.eval(); scf_tower.eval()
    # max_batches is a GLOBAL budget: the val set is sharded by DistributedSampler, so leaving it per-rank
    # would make the 2-GPU eval cover 2x the cells of the 1-GPU eval and break comparability with the
    # earlier single-GPU runs.
    max_batches = max(1, -(-max_batches // world))
    losses, n = [], 0
    hit = {("i", k): 0 for k in (1, 5, 10)} | {("g", k): 0 for k in (1, 5, 10)}
    ranks_i, ranks_g = [], []
    for b, (img, W, scf) in enumerate(loader):
        if b >= max_batches: break
        with torch.autocast("cuda", dtype=C.AMP_DTYPE):
            zi = C.he_embed(model, he_tower, img, W).float()
            zg = scf_tower(scf.to(C.dev, non_blocking=True)).float()
        losses.append(float(C.infonce(zi, zg)))
        logits = zi @ zg.t(); lab = torch.arange(len(zi), device=zi.device)
        # rank of the true partner, not just whether it won. R@1 throws away everything about the near
        # misses, so it is a binomial with se sqrt(p(1-p)/n); the rank distribution uses the whole ordering
        # and its median moves far less between evals.
        for tag, lg in (("i", logits), ("g", logits.t())):
            true = lg.gather(1, lab[:, None])
            rk = (lg > true).sum(1) + 1                      # 1 = correct partner is top-1
            (ranks_i if tag == "i" else ranks_g).append(rk.cpu().numpy())
            for k in (1, 5, 10):
                hit[(tag, k)] += int((rk <= k).sum())
        n += len(zi)
    model.train(); he_tower.train(); scf_tower.train()
    ri = np.concatenate(ranks_i) if ranks_i else np.array([1])
    rg = np.concatenate(ranks_g) if ranks_g else np.array([1])
    if world > 1:
        # every rank scored a different shard of the val set; sum the counts so the reported metric is over
        # the whole set rather than 1/world of it, and so all ranks agree (early stopping must not diverge)
        t = torch.tensor([float(np.sum(losses)), float(len(losses)), float(n)]
                         + [float(hit[(g, k)]) for g in ("i", "g") for k in (1, 5, 10)]
                         + [float(np.sum(ri)), float(np.sum(1.0 / ri)),
                            float(np.sum(rg)), float(np.sum(1.0 / rg))],
                        device=C.dev, dtype=torch.float64)   # float32 loses integers past 2^24; the rank
                        # sum over 390k cells at mean rank ~58 is 2.3e7, already past that
        dist.all_reduce(t, op=dist.ReduceOp.SUM); t = t.cpu().numpy()
        losses_sum, n_batch, n = t[0], t[1], int(t[2])
        for j, (g, k) in enumerate([(g, k) for g in ("i", "g") for k in (1, 5, 10)]):
            hit[(g, k)] = float(t[3 + j])          # np scalar otherwise -> not JSON serializable
        # t[9] / t[11] are SUMS of ranks, so dividing gives the MEAN rank, not the median -- a median cannot
        # be recovered from per-rank medians or sums at all. Mean rank is reported instead, and is the honest
        # name for what this number is; the rank distribution is heavy-tailed so the two differ a lot.
        out = {"val_infonce": float(losses_sum / max(n_batch, 1)), "val_cells": n,
               "mean_rank_he2scf": float(t[9] / max(n, 1)), "mean_rank_scf2he": float(t[11] / max(n, 1)),
               "mrr_he2scf": float(t[10] / max(n, 1)), "mrr_scf2he": float(t[12] / max(n, 1))}
        for k in (1, 5, 10):
            out[f"R@{k}_he2scf"] = float(hit[("i", k)]) / max(n, 1)
            out[f"R@{k}_scf2he"] = float(hit[("g", k)]) / max(n, 1)
        return out
    out = {"val_infonce": float(np.mean(losses)), "val_cells": n,
           "mean_rank_he2scf": float(np.mean(ri)), "mean_rank_scf2he": float(np.mean(rg)),
           "mrr_he2scf": float(np.mean(1.0 / ri)), "mrr_scf2he": float(np.mean(1.0 / rg))}
    for k in (1, 5, 10):
        out[f"R@{k}_he2scf"] = float(hit[("i", k)]) / max(n, 1)
        out[f"R@{k}_scf2he"] = float(hit[("g", k)]) / max(n, 1)
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", choices=["adamw"], required=True)
    ap.add_argument("--save_every", type=int, default=0,
                    help="also checkpoint every N steps, tagged with the step. Stage-1 in-batch\n                         retrieval is a PROXY: it peaked at 0.48 epoch on the 25M run while train\n                         loss kept falling, so selecting `best` on it discards a model that has\n                         seen every cell. Keeping the trajectory lets Stage 2/3 pick instead.")
    ap.add_argument("--target", choices=["scf", "scgpt"], default="scf",
                    help="alignment target: scFoundation (default, v25mfull) or scGPT cell embeddings "
                         "(needs VOICE_SCGPT_EMB, see scgpt_cellemb.py)")
    ap.add_argument("--platforms", default="xenium,cosmx",
                    help="comma-separated; both by default now that CosMx is embedded")
    ap.add_argument("--loss", choices=["infonce", "siglip", "dcl", "learntemp"], default="infonce",
                    help="Stage-1 contrastive objective. Training-loss VALUES are not comparable "
                         "across these; eval always uses fixed-temperature InfoNCE, so val_infonce "
                         "and R@k are.")
    ap.add_argument("--adjust_on", choices=["param", "block"], default="block",
                    help="shape setting the match_rms_adamw factor. block = step size matched to whole-tensor Muon (the clean granularity ablation); param = also inflates the step ~10x")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--max_cells", type=int, default=400000)
    ap.add_argument("--val_cells", type=int, default=20000)
    ap.add_argument("--eff_batch", type=int, default=2048); ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--lr_lora", type=float, default=1.5e-4); ap.add_argument("--lr_tower", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--nblocks", type=int, default=12); ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32); ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many consecutive evals with no improvement in EITHER val_infonce "
                         "or R@1 (0 = never stop early). Both must stall: the curves are jagged enough that "
                         "one metric alone dips on noise -- Muon+D4's R@1 argmax landed on step 500 in one "
                         "run and 3500 in another with identical settings.")
    ap.add_argument("--uni2h", default=UNI2H, help="local UNI2-h dir (config.json + pytorch_model.bin)")
    ap.add_argument("--d4_cycle", action="store_true",
                    help="cycle a per-cell permutation of D4 instead of drawing with replacement, so n epochs "
                         "give exactly min(n,8) distinct orientations per cell rather than 8*(1-(7/8)^n)")
    ap.add_argument("--d4_aug", action="store_true",
                    help="random D4 (rot90 x flip) on train crops, mask rotated with them; val stays canonical")
    ap.add_argument("--smoke", action="store_true", help="2 train / 1 val slide -- small scale, REAL weights")
    ap.add_argument('--epochs', type=int, default=None, help='derive optimizer steps from effective training cells')
    ap.add_argument('--retain_tail',action='store_true',help='supervise all usable cells, including masked DDP tail batches')
    ap.add_argument('--no_validation', action='store_true', help='fixed training budget; never read test_set')
    ap.add_argument('--resume', default='', help='restore weights/optimizer and skip completed sampler slots; same world size')
    a = ap.parse_args()

    global TARGET
    TARGET = a.target
    if TARGET == "scgpt":
        assert SCGPT_EMB and os.path.isdir(SCGPT_EMB), "--target scgpt needs VOICE_SCGPT_EMB=<dir of <obj>.npy>"
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0)); r0 = rank == 0
    if world > 1:
        torch.cuda.set_device(local)                       # BEFORE init: pins this rank to its GPU
        dist.init_process_group("nccl", timeout=timedelta(hours=4),
                                device_id=torch.device(f"cuda:{local}"))
        C.dev = torch.device(f"cuda:{local}")
    # identical on every rank and in every arm: the init_hash assertion depends on it
    torch.manual_seed(a.seed); np.random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    os.makedirs(CKDIR, exist_ok=True)

    # Both platforms. usable_slides defaults to xenium, and every run in this series before the scF
    # embeddings were completed used that default -- with CosMx unembedded there was nothing to include.
    # Now that all 101 objects are row-aligned, restricting to xenium would silently drop 13 training
    # objects (1,649,619 cells) and 4 val objects.
    plats = [x for x in a.platforms.split(",") if x]
    tr = [o for p_ in plats for o in usable_slides("training_set", p_)]
    from voice.training_contract import dataset_contract, check_members, stage1_steps
    from voice.training_resume import (Stage1Cursor, stage1_data_identity, check_stage1_resume,
        restore_stage1_weights, capture_rng_per_rank, restore_main_rng, atomic_torch_save, STAGE1_REVISION)
    run_contract = dataset_contract(DATASET)
    if run_contract is not None:
        if a.max_cells or not a.retain_tail or not a.epochs:
            raise ValueError('New Stage1 requires all cells, explicit epochs and retained tail batches')
        if a.smoke:
            raise ValueError('Use a separate legacy/synthetic dataset for smoke runs; do not label a partial run as dataset25m')
        if not a.no_validation:
            raise ValueError('New split requires --no_validation; cross-slide cannot be a validation pool')
        check_members(run_contract, 'stage1', [o for o,d in tr])
    va = [] if a.no_validation else [o for p_ in plats for o in usable_slides("test_set", p_)]
    if r0: print(f"[data] {len(tr)} train / {len(va)} val row-aligned objects over {plats} | target {TARGET}", flush=True)
    if a.smoke: tr, va = tr[:2], va[:1]
    paths, W, scf = build_data(tr, a.max_cells, a.seed)
    if a.no_validation:
        vpaths, vW, vscf = [], np.empty((0,16,16),np.float16), np.empty((0,scf.shape[1]),np.float16)
    else:
        vpaths, vW, vscf = build_data(va, a.val_cells, a.seed)
    print(f"[data] train {len(paths):,} cells | val {len(vpaths):,} cells", flush=True)

    model = build_uni2_local(C.dev, a.uni2h)
    C.inject_lora(model, a.nblocks, a.r, a.alpha, a.dropout)
    model.to(C.dev)

    mu, sd = C.scf_stats(scf)
    he_tower = C.HeTower().to(C.dev); scf_tower = C.ScfTower(mu, sd).to(C.dev)

    # The A/B is only a test of the optimizer if both arms START from the same weights. LoRA A/B and the two
    # towers are randomly initialised, so a difference in how much RNG was consumed before this point would
    # silently make the arms incomparable. Stamp a hash of every trainable tensor; the two runs must match.
    h = hashlib.sha256()
    for n, t in sorted(list(C.trainable(model).items())
                       + [("he." + k, v) for k, v in he_tower.state_dict().items()]
                       + [("scf." + k, v) for k, v in scf_tower.state_dict().items()]):
        h.update(n.encode()); h.update(t.detach().float().cpu().numpy().tobytes())
    init_hash = h.hexdigest()[:16]
    if r0: print(f"[init] trainable-weight hash = {init_hash}   (seeded initialization)", flush=True)

    lossfn = ContrastLoss(a.loss).to(C.dev)
    opt = build_optimizer(a.optimizer, model, he_tower, scf_tower, a.lr_lora, a.lr_tower, a.wd,
                          adjust_on=a.adjust_on, loss_params=list(lossfn.parameters()))
    if r0: print(f"[loss] {a.loss} ({sum(q.numel() for q in lossfn.parameters())} learnable scalars)", flush=True)

    g = torch.Generator(); g.manual_seed(a.seed)                        # identical data ORDER in both arms
    tds = C.CLIPDS(paths, W, scf)
    if a.d4_aug:
        tds = D4Wrap(tds, cycle=a.d4_cycle, seed=a.seed)
    # cycle mode needs the epoch to reach the workers, and persistent workers are never re-forked, so they
    # would keep serving epoch 0 forever. Recreating them each epoch costs a few seconds.
    persist = a.workers > 0 and not (a.d4_aug and a.d4_cycle)
    if a.retain_tail:
        from voice.clip_coverage import TailDataset,ResumeTailSampler
        tds=TailDataset(tds)
        samp=ResumeTailSampler(tds,num_replicas=world,rank=rank,shuffle=True,drop_last=False,seed=a.seed)
    else:
        samp = DistributedSampler(tds, num_replicas=world, rank=rank, shuffle=True,
                                  drop_last=True, seed=a.seed) if world > 1 else None
    dl = DataLoader(tds, batch_size=a.chunk, sampler=samp, shuffle=(samp is None), generator=g,
                    drop_last=not a.retain_tail, num_workers=a.workers, pin_memory=True, persistent_workers=persist)
    # build_data concatenates slides in usable_slides order, so an unshuffled val loader hands out one slide
    # at a time. evaluate() stops after max_batches, and 200 x 256 = 51,200 cells is less than the first
    # slide holds -- every val number would come from breast_cancer alone, with kidney and pancreas never
    # scored. A fixed permutation (same every eval, every arm, every rank, so the metric stays comparable)
    # makes the scored prefix a random sample of the whole split instead.
    vperm = np.random.RandomState(12345).permutation(len(vpaths))
    vpaths = [vpaths[i] for i in vperm]; vW = vW[vperm]; vscf = vscf[vperm]
    vds = C.CLIPDS(vpaths, vW, vscf)
    vsamp = DistributedSampler(vds, num_replicas=world, rank=rank, shuffle=False,
                               drop_last=True) if world > 1 else None
    vdl = DataLoader(vds, batch_size=a.chunk, sampler=vsamp, shuffle=False, drop_last=True,
                     num_workers=max(2, a.workers // 2), pin_memory=True)
    # a second val pass on DETERMINISTICALLY rotated crops. Canonical val answers "did D4 help the orientation
    # the model is scored on"; this one answers "did D4 buy rotation robustness", which is the thing D4 is for.
    # It is reported by every arm, so the augmented and un-augmented arms stay comparable on both axes.
    vrsamp = DistributedSampler(D4Wrap(vds, fixed=True), num_replicas=world, rank=rank,
                                shuffle=False, drop_last=True) if world > 1 else None
    vdl_rot = DataLoader(D4Wrap(vds, fixed=True), batch_size=a.chunk, sampler=vrsamp, shuffle=False,
                         drop_last=True, num_workers=max(2, a.workers // 2), pin_memory=True)

    n_chunk = max(1, a.eff_batch // a.chunk)
    if a.epochs is not None:
        if a.epochs < 1 or (len(dl) < n_chunk and not a.retain_tail):
            raise ValueError('Positive epochs and at least one full optimizer batch are required')
        a.steps = stage1_steps(len(tds),a.epochs,world,a.chunk,n_chunk,retain_tail=a.retain_tail)
    hist = []; step = 0; t0 = time.time(); buf = []; epoch = 0
    data_identity=stage1_data_identity(tr,paths,os.environ.get('VOICE_TRAIN_CACHE')) if a.retain_tail else None
    resumed = None
    resume_cursor = None
    if a.resume:
        resumed=torch.load(a.resume,map_location='cpu',weights_only=False)
        resume_cursor=check_stage1_resume(resumed,a,run_contract,world,len(tds),data_identity,init_hash)
        restore_stage1_weights(resumed,model,he_tower,scf_tower,opt,lossfn)
        step=resumed['step'];epoch=resume_cursor.epoch;hist=resumed.get('history',[])
        if 'loader_generator_state' in resumed:g.set_state(resumed['loader_generator_state'].cpu())
        if r0:
            print(f'[resume] {a.resume}: step {step}/{a.steps}; epoch {epoch+1}; '
                  f'skip {resume_cursor.skip_batches:,} batches / {resume_cursor.skip_slots:,} '
                  'sampler slots per rank without decoding their PNGs; optimizer restored',flush=True)
            if 'rng_per_rank' not in resumed:
                print('[resume] old checkpoint has no RNG states; D4/dropout trajectory changes, '
                      'while optimizer and target coverage continue',flush=True)
            elif a.workers:
                print('[resume] main-process RNG will be restored; worker D4 randomness may change',flush=True)
    def saved_state(rng):
        cursor=Stage1Cursor(len(tds),world,a.chunk,n_chunk,a.epochs,step) if a.retain_tail and a.epochs else None
        return {'lora':C.trainable(model),'he_tower':he_tower.state_dict(),
                'scf_tower':scf_tower.state_dict(),'lossfn':lossfn.state_dict(),
                'args':vars(a),'init_hash':init_hash,'history':hist,'step':step,
                'world_size':world,'run_contract':run_contract,'opt':opt.state_dict(),
                'resume_revision':STAGE1_REVISION,'data_identity':data_identity,
                'coverage_cursor':cursor.record() if cursor else None,
                'rng_per_rank':rng,'loader_generator_state':g.get_state(),
                'worker_rng_captured':False}
    best_vi, best_r1, best_step, stall, stop = float('inf'), -1.0, 0, 0, False
    if r0: print(f"[run] {a.optimizer} world={world} tag={a.tag} steps={a.steps} eff_batch={a.eff_batch} "
          f"({n_chunk} x {a.chunk}) lr_lora={a.lr_lora} lr_tower={a.lr_tower} wd={a.wd}", flush=True)
    while step < a.steps and not stop:
        if samp is not None:
            samp.set_epoch(epoch)                          # otherwise every epoch reuses one permutation
        skip_slots=resume_cursor.skip_slots if resume_cursor is not None and epoch==resume_cursor.epoch else 0
        if a.retain_tail:samp.set_start(skip_slots)
        if a.d4_aug and a.d4_cycle:
            (tds.base if a.retain_tail else tds).set_epoch(epoch)
        supervised_epoch=resume_cursor.supervised_in_epoch(rank) if skip_slots else 0
        iterator=iter(dl)
        if resumed is not None:
            restore_main_rng(resumed,rank,world)
            resumed=None
        for batch_index,batch in enumerate(iterator):
            supervised_epoch+=int(batch[3].sum()) if a.retain_tail else len(batch[0])
            buf.append(batch)
            if len(buf) < n_chunk and not (a.retain_tail and batch_index==len(dl)-1): continue
            opt.set_lr_scale(min(1.0, (step + 1) / max(a.warmup, 1)))
            loss = gradcache_step_ddp(buf, model, he_tower, scf_tower, opt, world, lossfn); buf = []; step += 1
            if step % 25 == 0:
                print(f"  step {step:>5d}/{a.steps} loss {loss:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if not a.no_validation and (step % a.eval_every == 0 or step == a.steps):
                ev = evaluate(model, he_tower, scf_tower, vdl, world=world)
                rot = evaluate(model, he_tower, scf_tower, vdl_rot, world=world)
                ev.update({f"rot_{k2}": v2 for k2, v2 in rot.items() if k2 != "val_cells"})
                ev.update(step=step, train_loss=loss,
                          **{f"loss_{n2}": float(v2) for n2, v2 in lossfn.named_parameters()})
                hist.append(ev)
                if r0: print(f"  [eval] step {step:>5d} infonce {ev['val_infonce']:.4f} | he->scf "
                      f"R@1 {ev['R@1_he2scf']:.4f} R@5 {ev['R@5_he2scf']:.4f} R@10 {ev['R@10_he2scf']:.4f} "
                      f"meanRank {ev['mean_rank_he2scf']:.1f} MRR {ev['mrr_he2scf']:.4f} | rot R@1 "
                      f"{ev['rot_R@1_he2scf']:.4f} | n={ev['val_cells']:,}", flush=True)
                if r0:
                  with open(f"{CKDIR}/{a.tag}_history.json", "w") as f:
                    json.dump({"args": vars(a), "init_hash": init_hash, "history": hist}, f, indent=2, default=float)

                if r0 and a.save_every and step % a.save_every == 0:
                    # opt state included so these are RESUMABLE. Without the AdamW moments, restarting from
                    # a checkpoint zeroes the momentum and re-runs warmup: a different trajectory, not a
                    # continuation. The 25M run's checkpoints lack it, which is why its rerun had to start
                    # from scratch and pay 9h of recompute. Costs ~2x the parameter bytes (~35 MB here).
                    torch.save({"lora": C.trainable(model), "he_tower": he_tower.state_dict(),
                                "scf_tower": scf_tower.state_dict(), "args": vars(a),
                                "init_hash": init_hash, "step": step, "eval": ev,
                                "opt": opt.state_dict()},
                               f"{CKDIR}/{a.tag}_step{step}.pt")

                # best-so-far checkpoint, because with early stopping the last step is not the best one
                # compare smoothed values: a single spike should neither reset the patience counter nor
                # be recorded as the best point
                k_sm = 3
                sm_vi = float(np.mean([e["val_infonce"] for e in hist[-k_sm:]]))
                sm_r1 = float(np.mean([e["R@1_he2scf"] for e in hist[-k_sm:]]))
                if sm_vi < best_vi - 1e-6 or sm_r1 > best_r1 + 1e-6:
                    if sm_vi < best_vi: best_vi, best_step = sm_vi, step
                    if sm_r1 > best_r1: best_r1 = sm_r1
                    stall = 0
                    if r0:
                        torch.save({"lora": C.trainable(model), "he_tower": he_tower.state_dict(),
                                    "scf_tower": scf_tower.state_dict(), "args": vars(a),
                                    "init_hash": init_hash, "step": step, "eval": ev},
                                   f"{CKDIR}/{a.tag}_best.pt")
                else:
                    stall += 1
                    if r0: print(f"         no improvement in either metric ({stall}"
                          + (f"/{a.patience}" if a.patience else "") + ")", flush=True)
                    if a.patience and stall >= a.patience:
                        if r0: print(f"  [early stop] {a.patience} evals without improvement; best was "
                              f"infonce {best_vi:.4f} / R@1 {best_r1:.4f} around step {best_step}", flush=True)
                        stop = True
                        break
            if a.no_validation and ((a.save_every and step % a.save_every == 0) or step == a.steps):
                rng=capture_rng_per_rank()
                if r0:atomic_torch_save(saved_state(rng),f'{CKDIR}/{a.tag}_step{step}.pt')
            if step >= a.steps: break
        if a.epochs is not None:
            buf = []
        if a.retain_tail:
            seen=torch.tensor(supervised_epoch,device=C.dev,dtype=torch.long)
            if world>1:dist.all_reduce(seen)
            if seen.item()!=len(tds):raise ValueError('Stage1 epoch coverage mismatch')
            if r0:print(f'[coverage] Stage1 epoch {epoch+1}: {seen.item():,} unique supervised cells',flush=True)
        epoch += 1
        resume_cursor=None

    if resumed is not None:
        # A checkpoint already at the fixed budget does not enter the training loop.
        restore_main_rng(resumed,rank,world)
    rng=capture_rng_per_rank()
    if r0:atomic_torch_save(saved_state(rng),f'{CKDIR}/{a.tag}_final.pt')
    if world > 1: dist.destroy_process_group()
    if r0: print(f"[done] {a.tag} -> {CKDIR}/{a.tag}_final.pt  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
