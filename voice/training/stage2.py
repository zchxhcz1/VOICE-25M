"""Train the image encoder, spatial decoder and expression head."""
from __future__ import annotations
import os, sys, json, time, argparse, types, glob, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, scipy.sparse as sp, torch
import torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import stage1 as M
from voice.training_contract import dataset_contract, check_members, check_checkpoint, gene_hash, checkpoint_identity, load_lora_checked, check_stage1_identity, stage2_steps
from voice.tissue import object_directory
from voice.full_coverage import CoveragePlan, FullCoverageDataset, supervised_sums, REVISION as COVERAGE_REVISION
C = M.C
from expression_head import DirectHead, nb_nll, cell_areas
from spatial import LoopedSE2Transformer
from cell_inputs import cells_of

R = os.environ.get("VOICE_ROOT", "runs")
FEAT = os.environ.get("VOICE_FEAT", "runs/features")
CKDIR = f"{R}/checkpoints/expression"
SPACE = f"{R}/direct/gene_space_all.npz"



class LiveStore:
    """One object: crop paths, 16x16 cmask weights, micron positions, panel counts, areas, kNN sampler.

    Same geometry as FrozenStore, but the per-cell payload is a PNG path instead of a 1536-d vector. Only the
    masks are held in memory (1 KB per cell), so `--max_cells` exists to bound that as well as to bound how
    much of a slide one run sees.
    """

    def __init__(self, obj, d, plat, space, aux_flag, n_nb, max_cells=0):
        cm = f"{FEAT}/{'cosmx' if plat else 'xenium'}/{obj}/cmask.npz"
        if os.path.exists(cm):
            z = np.load(cm)
            W = z["W"].astype(np.float32); er = z["expr_rows"]; pos_px = z["pos_px"]
            man = (pd.read_csv(f"{d}/manifest.csv.gz", usecols=["expr_row", "patch_path"])
                   .drop_duplicates("expr_row").set_index("expr_row")["patch_path"])
            paths = [os.path.join(d, str(man.loc[int(e)])) for e in er]
            W = W / np.clip(W.sum((1, 2), keepdims=True), 1e-8, None)
        else:
            paths, W, er, pos_px = cells_of(d)
        ums = float(json.load(open(f"{d}/dataset.json"))["um_per_pixel_source"])
        pos = pos_px * ums                                            # -> microns
        rows = np.arange(len(er))          # positions into features.npz; narrowed by max_cells below
        if max_cells and len(er) > max_cells:
            lo = pos.min(0); span = pos.max(0) - lo
            side = float(np.sqrt(max_cells / len(er)) * np.sqrt(span[0] * span[1]))
            rng = np.random.RandomState(0); best = None
            for _ in range(40):
                l = lo + rng.rand(2) * np.maximum(span - side, 0)
                k = np.where(((pos >= l) & (pos <= l + side)).all(1))[0]
                if best is None or len(k) > len(best):
                    best = k
            keep = best
            paths = [paths[i] for i in keep]; W = W[keep]; er = er[keep]; pos = pos[keep]
            rows = keep
        cache=os.environ.get('VOICE_TRAIN_CACHE')
        if cache:
            from voice.training_cache import cached_arrays, file_stamp, array_stamp
            signature={'boundaries':file_stamp(f'{d}/patch_cell_boundaries.npz'),
                       'mask':file_stamp(cm) if os.path.exists(cm) else None,
                       'rows':array_stamp(er),'max_cells':max_cells,'schema':1}
            def build_masks(root):
                np.save(root/'W.tmp.npy',W);return ['W']
            W=cached_arrays(f'{cache}/stage2/{obj}/masks',signature,build_masks)['W']
        self.paths, self.W, self.pos = np.array(paths, dtype=np.bytes_), W, pos
        self._t = torch.from_numpy(pos)
        self.rows = np.asarray(rows, np.int64)
        self.expr_rows = np.asarray(er, np.int64)

        X = sp.load_npz(f"{d}/expression.npz")
        syms = pd.read_csv(f"{d}/genes.tsv", sep="\t").iloc[:, 0].astype(str).tolist()
        head = {g: i for i, g in enumerate(space["genes"].tolist())}
        cols, hidx, seen = [], [], set()
        for j, g in enumerate(syms):
            if g in head and g not in seen:
                seen.add(g); cols.append(j); hidx.append(head[g])
        self.head_idx = np.asarray(hidx, np.int64)
        self.is_aux = np.asarray(aux_flag[self.head_idx], bool)
        if cache:
            from voice.training_cache import sparse_targets
            self.y=sparse_targets(f'{cache}/stage2/{obj}/targets',X,er,np.asarray(cols,np.int64),
                                  {'expression':file_stamp(f'{d}/expression.npz'),
                                   'genes':file_stamp(f'{d}/genes.tsv'),'schema':1})
        else:
            self.y = X[er][:, np.asarray(cols, np.int64)].toarray().astype(np.float16)
        area, med = cell_areas(d, er)
        self.area_abs = np.log(area + 1.0).astype(np.float32)
        self.area_rel = (np.log(area + 1.0) - np.log(med + 1.0)).astype(np.float32)
        self.plat, self.obj, self.n, self.n_nb = int(plat), obj, len(er), n_nb

    def neighbourhood(self, anchor):
        d = (self._t - self._t[anchor]).pow(2).sum(1)
        return torch.topk(d, min(self.n_nb, self.n), largest=False).indices.numpy()

    def load(self, idx, rot=0, flip=False):
        """Decode the neighbourhood's crops. One orientation for the whole scene (see module docstring)."""
        img = np.stack([np.asarray(Image.open(self.paths[i].decode()).convert("RGB").resize((224, 224), Image.BILINEAR),
                                   np.uint8).transpose(2, 0, 1) for i in idx])
        W = self.W[idx]
        if rot:
            img = np.rot90(img, rot, axes=(2, 3)).copy(); W = np.rot90(W, rot, axes=(1, 2)).copy()
        if flip:
            img = img[:, :, :, ::-1].copy(); W = W[:, :, ::-1].copy()
        return img, W


class NbDataset(Dataset):
    """Yields whole neighbourhoods. Length is the step budget; each item samples its own anchor.

    Objects are drawn proportional to cell count so a step is a uniform sample over cells, not over slides.
    Seeding is per (rank, worker, index) so no two workers or ranks walk the same sequence.
    """

    def __init__(self, stores, steps, seed, rank, d4, start=0):
        self.st, self.steps, self.seed, self.rank, self.d4 = stores, steps, seed, rank, d4
        self.start = start
        w = np.array([s.n for s in stores], float); self.w = w / w.sum()

    def __len__(self):
        return self.steps

    def __getitem__(self, i):
        i += self.start
        rng = np.random.RandomState((self.seed * 1000003 + self.rank * 10007 + i) % (2 ** 31 - 1))
        s = self.st[rng.choice(len(self.st), p=self.w)]
        idx = s.neighbourhood(rng.randint(s.n))
        rot, flip = (rng.randint(4), bool(rng.randint(2))) if self.d4 else (0, False)
        img, W = s.load(idx, rot, flip)
        return (torch.from_numpy(img), torch.from_numpy(W),
                torch.from_numpy(s.y[idx].astype(np.float32)),
                torch.from_numpy(s.area_abs[idx]), torch.from_numpy(s.area_rel[idx]),
                torch.from_numpy(s.pos[idx]), int(s.plat), torch.from_numpy(s.head_idx),
                torch.from_numpy(np.where(s.is_aux, 1, 0).astype(np.float32)),torch.ones(len(idx),dtype=torch.bool))


def one(b):
    return b[0]                                       # batch_size=1: a neighbourhood is already the batch



class SE2Live(nn.Module):
    def __init__(self, enc, gene_emb, d_model=256, n_layers=6, span=0.0, r=1, max_dist=137.0, **head_kw):
        super().__init__()
        self.enc = enc
        self.inp = nn.Linear(1536, d_model)
        self.se2 = LoopedSE2Transformer(d_model, 64, 8, n_layers=n_layers, span=span, r=r, max_dist=max_dist)
        self.head = DirectHead(gene_emb, d_in=d_model, **head_kw)

    def forward(self, img_u8, W, pos, plat, aa, ar, hi, neutral=False):
        with torch.autocast("cuda", dtype=C.AMP_DTYPE):
            h = C.pooled_feat(self.enc, C.norm(img_u8), W)
        return self.head(self.se2(self.inp(h.float()), pos), plat, aa, ar, hi, neutral=neutral)


GENE_EMB_SCGPT = f"{R}/direct/gene_emb_scgpt.npz"


def gene_table(space, which="scf"):
    """The [G, D] gene table the head reads (frozen) or is initialised from (--free_head).

    scf    scFoundation pos_emb rows, stored in gene_space_all.npz -- live_nod4_c's table (D = 768)
    scgpt  scGPT encoder.embedding + enc_norm, built by build_scgpt_gene_emb.py (D = 512); the 120 genes outside
           scGPT's vocabulary have zero rows, which a --free_head run learns from data like any other row
    <path> an .npz with `genes` (same order as gene_space_all) and `emb`
    """
    if which == "scf":
        return np.asarray(space["gene_emb"], np.float32)
    p = GENE_EMB_SCGPT if which == "scgpt" else which
    z = np.load(p, allow_pickle=True)
    assert z["genes"].tolist() == space["genes"].tolist(), f"{p}: gene order differs from gene_space_all"
    return z["emb"].astype(np.float32)


def build_from_ckpt(enc, space, ck):
    """SE2Live shaped for a Stage-2 checkpoint (weights NOT loaded -- the caller loads inp/se2/head).

    The read-out is inferred from the saved head, so this serves checkpoints from before and after the
    --gene_emb/--free_head flags: E is a persistent buffer of the saved width (the table itself is restored by
    head.load_state_dict), and a saved W means a learnable read-out.
    """
    ta = dict(ck["args"])
    contract = dataset_contract(M.DATASET)
    check_checkpoint(ck, contract, space['genes'].tolist())
    if contract is not None:
        check_stage1_identity(ck)
    E = ck["head"]["E"]
    return SE2Live(enc, np.zeros(tuple(E.shape), np.float32), d_model=ta["d_model"], n_layers=ta["n_layers"],
                   span=ta["span"], r=ta["loop_r"], max_dist=ta["max_dist"], use_plat_in=False,
                   use_plat_main=False, use_plat_gene=False, learn_gamma=True, free_head="W" in ck["head"])


def freeze_first_lora(enc, nblocks, frac):
    """Freeze LoRA in the first ceil(frac*nblocks) of the nblocks adapted blocks -- VOICE's own rule
    (train/train_phase2.py:freeze_first_lora). The early blocks carry generic texture; letting the gene loss
    move them is where a 25M-cell Phase-1 initialisation would be thrown away fastest."""
    n = len(enc.blocks); nfz = int(math.ceil(frac * nblocks)); fz = 0
    for i in range(n - nblocks, n - nblocks + nfz):
        for mod in (enc.blocks[i].attn.qkv, enc.blocks[i].attn.proj):
            for p in (mod.A, mod.B):
                p.requires_grad_(False); fz += 1
    return nfz, fz



class EvalDS(Dataset):
    """(store, anchor) pairs for evaluation, decoded by worker processes.

    Eval used to decode its PNGs synchronously in the main process while training had eight workers
    prefetching, which made an eval neighbourhood take ~1 s against training's 234 ms -- 8.9 minutes per eval,
    and the other ranks idle at the barrier throughout.

    The `seen` dedup that follows is order-dependent, but the neighbourhood itself is not: `idx` comes from the
    anchor alone. So the decode parallelises and the bookkeeping stays sequential in the main loop. Items are
    emitted store-major in anchor order and DataLoader preserves that order, so num_workers=0 reproduces the
    original loop exactly and num_workers=8 gives the same numbers faster.
    """

    def __init__(self, stores, n_anchor):
        self.st = stores
        self.items = [(si, int(a)) for si, s in enumerate(stores)
                      for a in np.linspace(0, s.n - 1, n_anchor).astype(int)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        si, a = self.items[k]
        s = self.st[si]
        idx = s.neighbourhood(a)
        img, W = s.load(idx)                       # canonical orientation: no D4 at eval
        return si, torch.from_numpy(img), torch.from_numpy(W), torch.from_numpy(idx)


@torch.no_grad()
def evaluate(model, stores, dev, n_anchor, workers=8, hvg_k=(50, 200)):
    core = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    core.eval()
    acc = [{"P": [], "Y": [], "seen": np.zeros(s.n, bool)} for s in stores]
    hi_of = [torch.from_numpy(s.head_idx).to(dev) for s in stores]
    dl = DataLoader(EvalDS(stores, n_anchor), batch_size=1, num_workers=workers,
                    collate_fn=one, pin_memory=True,
                    prefetch_factor=4 if workers else None)
    for si, img, W, idx in dl:
        st = stores[si]; a = acc[si]; ix = idx.numpy()
        keep = ~a["seen"][ix]
        if not keep.any():
            continue
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)
        plat = torch.full((len(ix),), st.plat, dtype=torch.long, device=dev)
        mu = core(img.to(dev), W.to(dev), t(st.pos[ix]), plat,
                  t(st.area_abs[ix]), t(st.area_rel[ix]), hi_of[si])
        k = torch.from_numpy(np.where(keep)[0]).to(dev)
        a["P"].append(torch.log1p(mu[k]).float().cpu())
        a["Y"].append(torch.log1p(t(st.y[ix].astype(np.float32))[k]).float().cpu())
        a["seen"][ix[keep]] = True
    core.train()
    out = {}
    for st, a in zip(stores, acc):
        Y = torch.cat(a["Y"]).numpy(); v = Y.std(0); good = v > 1e-8; y = Y[:, good]
        p = torch.cat(a["P"]).numpy()[:, good]
        pc = np.nan_to_num(((p - p.mean(0)) * (y - y.mean(0))).mean(0) / (p.std(0) * y.std(0) + 1e-12))
        aux = st.is_aux[good]; o = np.argsort(-v[good])
        r = {"n_cells": int(len(Y)), "n_genes": int(good.sum()), "platform": st.plat,
             "pearson_mean": float(pc.mean()),
             "pearson_primary": float(pc[~aux].mean()) if (~aux).any() else float("nan"),
             "pearson_aux": float(pc[aux].mean()) if aux.any() else float("nan")}
        for k_ in hvg_k:
            r[f"pearson_hvg{k_}"] = float(pc[o[:k_]].mean())
        out[st.obj] = r
    keys = ("pearson_mean", "pearson_primary", "pearson_aux", "pearson_hvg50", "pearson_hvg200")
    agg = {}
    for pl, nm in ((0, "xenium"), (1, "cosmx")):
        v = [x for x in out.values() if x["platform"] == pl]
        if v:
            agg[nm] = {k: float(np.nanmean([x[k] for x in v])) for k in keys}
    agg["all"] = {k: float(np.nanmean([x[k] for x in out.values()])) for k in keys}
    return agg, out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="live_se6")
    ap.add_argument("--train_objects", required=True)
    ap.add_argument("--val_objects", default='')
    ap.add_argument('--no_validation',action='store_true',help='fixed epochs; no validation labels or model selection')
    ap.add_argument('--epochs',type=int,default=None,help='global neighbourhood budget = epochs * effective cells / n_nb')
    ap.add_argument('--sampling',choices=['neighbourhood','full_coverage'],default='neighbourhood')
    ap.add_argument('--target_cells',type=int,default=128,help='maximum supervised cells per spatial block')
    ap.add_argument('--context_cells',type=int,default=64,help='maximum additional nearby context cells, excluded from loss')
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--n_anchor", type=int, default=100)
    ap.add_argument("--n_nb", type=int, default=128)
    ap.add_argument("--max_cells", type=int, default=400000)
    ap.add_argument("--max_dist", type=float, default=137.0)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--span", type=float, default=0.0)
    ap.add_argument("--loop_r", type=int, default=1)
    ap.add_argument("--lora_ckpt", default=f"{R}/checkpoints/alignment/stage1_final.pt")
    ap.add_argument("--nblocks", type=int, default=12)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--freeze_frac", type=float, default=0.2)
    ap.add_argument("--lr_lora", type=float, default=3e-5)
    ap.add_argument("--lr_se2", type=float, default=1e-4)
    ap.add_argument("--frozen_enc", action="store_true", help="control: no LoRA gradient at all")
    ap.add_argument("--grad_ckpt", action="store_true", help="ViT activation checkpointing (~30%% slower, much less memory)")
    ap.add_argument("--d4", type=int, default=1)
    ap.add_argument("--lam_aux", type=float, default=1.0)
    ap.add_argument("--lam_mse", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", default="", help="restore weights/optimizer and finish the remaining fixed-epoch steps; same world size")
    ap.add_argument("--gene_emb", default="scf", help="scf | scgpt | path to an .npz table (see gene_table)")
    ap.add_argument("--free_head", action="store_true", help="learnable gene rows instead of a frozen table")
    ap.add_argument("--free_init", choices=["emb", "default"], default="emb",
                    help="with --free_head: start from the --gene_emb table (emb) or nn.Linear's init (default)")
    a = ap.parse_args()
    from voice.training_resume import (check_stage2_resume, capture_rng_per_rank,
                                      restore_main_rng, atomic_torch_save)
    run_contract = dataset_contract(M.DATASET)
    if run_contract is not None:
        if a.sampling!='full_coverage' or not a.epochs or a.max_cells:
            raise ValueError('New dataset training requires full coverage, explicit epochs and max_cells=0')
        if not a.no_validation or a.val_objects:
            raise ValueError('New split uses fixed epochs without validation')
        check_members(run_contract,'stage2',[o for o in a.train_objects.split(',') if o])

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda")
    if run_contract is not None:
        torch.manual_seed(a.seed); np.random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    log = (lambda *x, **k: print(*x, **{**k, "flush": True})) if rank == 0 else (lambda *x, **k: None)
    os.makedirs(CKDIR, exist_ok=True)

    space = np.load(SPACE, allow_pickle=True)
    xen = set(np.load(f"{R}/direct/gene_space_xen.npz", allow_pickle=True)["genes"].tolist())
    aux_flag = np.array([g not in xen for g in space["genes"].tolist()], bool)
    dir_of, plat_of = {}, {}
    for i, o in enumerate(space["objects"].tolist()):
        try:
            d = object_directory(M.DATASET,o)
        except FileNotFoundError:
            continue
        dir_of[o] = str(d); plat_of[o] = int(space['platform'][i])

    def build(names):
        out = []
        for o in [x for x in names.split(",") if x]:
            if o not in dir_of:
                if run_contract is not None:
                    raise ValueError(f'{o}: missing gene-space object/dataset source')
                log(f"  [skip] {o}: not in the dataset"); continue
            try:
                s = LiveStore(o, dir_of[o], plat_of[o], space, aux_flag, a.n_nb, a.max_cells)
            except (FileNotFoundError, OSError) as e:      # the shared corpus does change under us
                if run_contract is not None:
                    raise
                log(f"  [MISSING] {o}: {type(e).__name__} -- excluded") if 'log' in dir() else \
                    print(f"  [MISSING] {o}: {type(e).__name__} -- excluded", flush=True)
                continue
            out.append(s)
            log(f"  [{o[:30]:30s}] {s.n:>9,} cells {len(s.head_idx):>7,} genes  plat={s.plat}")
        return out

    tr, va = build(a.train_objects), ([] if a.no_validation else build(a.val_objects))
    assert tr and (a.no_validation or va)
    coverage=None
    if a.sampling=='full_coverage':
        if not a.epochs or not a.no_validation or a.target_cells<1 or a.context_cells<0:
            raise ValueError('Full coverage requires fixed epochs, no validation and valid block sizes')
        coverage=CoveragePlan(tr,a.epochs,a.target_cells,a.seed,world,os.environ.get('VOICE_TRAIN_CACHE'))
        a.steps=coverage.blocks
        log(f'[coverage] {coverage.cells:,} target cells each epoch; {coverage.blocks:,} blocks; '
            f'{coverage.total_steps:,} steps/rank; max core/context={a.target_cells}/{a.context_cells}')
    elif a.epochs is not None:
        if a.epochs < 1:
            raise ValueError('epochs must be positive')
        a.steps, _ = stage2_steps(sum(s.n for s in tr),a.epochs,a.n_nb,world)
    log(f"[data] train {sum(s.n for s in tr):,} cells / {len(tr)} | val {sum(s.n for s in va):,} / {len(va)}")

    enc = M.build_uni2_local(dev)
    C.inject_lora(enc, a.nblocks, a.lora_r, a.lora_alpha, a.lora_dropout)
    ck = torch.load(a.lora_ckpt, map_location="cpu", weights_only=False)
    check_checkpoint(ck,run_contract)
    if run_contract is not None and ck['args'].get('target')!='scgpt':
        raise ValueError('This run requires the new scGPT-aligned Stage1 checkpoint')
    if run_contract is not None and (not ck['args'].get('no_validation') or
                                    not ck['args'].get('epochs') or ck.get('step')!=ck['args'].get('steps')):
        raise ValueError('Stage1 must finish its fixed epoch budget before initializing Stage2')
    load_lora_checked(enc,ck['lora'],complete=True)
    stage1_identity = checkpoint_identity(a.lora_ckpt) if run_contract is not None else None
    for n_, p in enc.named_parameters():
        p.requires_grad_((".A" in n_ or ".B" in n_) and not a.frozen_enc)
    nfz = fz = 0
    if not a.frozen_enc and a.freeze_frac > 0:
        nfz, fz = freeze_first_lora(enc, a.nblocks, a.freeze_frac)
    if a.grad_ckpt and hasattr(enc, "set_grad_checkpointing"):
        enc.set_grad_checkpointing(True)
    log(f"[enc] {os.path.basename(a.lora_ckpt)} | LoRA on last {a.nblocks} blocks, first {nfz} frozen "
        f"({fz} tensors) | trainable enc params {sum(p.numel() for p in enc.parameters() if p.requires_grad):,}")

    table = gene_table(space, a.gene_emb)
    model = SE2Live(enc, table, d_model=a.d_model, n_layers=a.n_layers, span=a.span,
                    r=a.loop_r, max_dist=a.max_dist, use_plat_in=False, use_plat_main=False,
                    use_plat_gene=False, learn_gamma=True, free_head=a.free_head, free_init=a.free_init).to(dev)
    log(f"[se2] {model.se2.describe()}")
    log(f"[head] gene table {a.gene_emb} {tuple(table.shape)} | "
        + (f"learnable rows, init {a.free_init}" if a.free_head else "frozen rows"))
    rk = None
    if a.resume:
        rk = torch.load(a.resume, map_location="cpu", weights_only=False)
        check_checkpoint(rk,run_contract,space['genes'].tolist())
        if run_contract is not None:
            check_stage2_resume(rk,a,run_contract,space['genes'].tolist(),stage1_identity,world,coverage)
            expected={name for name,p in enc.named_parameters() if p.requires_grad}
            if set(rk['lora'])!=expected:
                raise ValueError('Resume checkpoint lacks trained LoRA tensors')
        load_lora_checked(enc,rk['lora'])
        model.inp.load_state_dict(rk["inp"]); model.se2.load_state_dict(rk["se2"])
        model.head.load_state_dict(rk["head"])
        log(f"[resume] {os.path.basename(a.resume)} @ step {rk['step']:,} "
            f"({len(rk['lora'])} LoRA tensors)"
            + ("  + optimizer state" if "opt" in rk else "  WITHOUT optimizer state -- moments restart"))
        if run_contract is not None and 'rng_per_rank' not in rk:
            log('[resume] old checkpoint has no RNG states; stochastic trajectory changes')

    lora_p = [p for p in enc.parameters() if p.requires_grad]
    rest_p = [p for n_, p in model.named_parameters() if p.requires_grad and not n_.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": lora_p, "lr": a.lr_lora},
                             {"params": rest_p, "lr": a.lr_se2}], weight_decay=0.0)
    if rk is not None and "opt" in rk:
        opt.load_state_dict(rk["opt"])
    log(f"[opt] lora {sum(p.numel() for p in lora_p):,} @ {a.lr_lora} | rest "
        f"{sum(p.numel() for p in rest_p):,} @ {a.lr_se2} | D4 {'on' if a.d4 else 'off'}")

    net = model
    if ddp:
        net = nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])],
                                                  gradient_as_bucket_view=True, broadcast_buffers=False)

    continuation = run_contract is not None or a.epochs is not None
    start_step = int(rk['step']) if rk is not None and continuation else 0
    total_steps = coverage.total_steps if coverage is not None else -(-a.steps//world)
    if start_step > total_steps:
        raise ValueError('Resume checkpoint exceeds the requested epoch budget')
    if start_step == total_steps:
        log('[done] requested epoch budget already completed')
        if ddp: dist.destroy_process_group()
        return
    sampler_seed = a.seed + (7919 if a.resume and not continuation else 0)
    if coverage is not None:
        ds=FullCoverageDataset(coverage,rank,a.context_cells,a.max_dist,a.seed,bool(a.d4),start=start_step)
        supervised=coverage.supervised_before(start_step,rank)
        if rk is not None and (rk.get('coverage_revision')!=COVERAGE_REVISION or
                               rk.get('supervised_cells_per_rank',[None]*world)[rank]!=supervised):
            raise ValueError('Resume coverage plan/count differs from the checkpoint')
    else:
        ds = NbDataset(tr,total_steps-start_step,sampler_seed,rank,bool(a.d4),start=start_step)
        supervised=0
    dl = DataLoader(ds, batch_size=1, num_workers=a.workers, collate_fn=one,
                    pin_memory=True, persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)

    hist, t0, best, best_step = [], time.time(), -1.0, -1
    if rank == 0 and not a.no_validation:          # step 0 sanity check on an explicit development set
        agg, per = evaluate(net, va, dev, a.n_anchor, a.eval_workers)
        b = agg["all"]
        log(f"  [eval] step      0 primary {b['pearson_primary']:.4f} aux {b['pearson_aux']:.4f} "
            f"hvg50 {b['pearson_hvg50']:.4f}   <- random trunk/head, init sanity check only "
            f"(NOT the frozen baseline -- that is --frozen_enc)")
        hist.append({"step": 0, "agg": agg, "per_object": per})
    if ddp:
        dist.barrier()

    step = start_step
    iterator=iter(dl)
    if rk is not None and continuation:restore_main_rng(rk,rank,world)
    for img, W, y, aa, ar, pos, plat, hi, isaux, target_mask in iterator:
        step += 1
        img = img.to(dev, non_blocking=True); W = W.to(dev, non_blocking=True)
        y = y.to(dev, non_blocking=True); aa = aa.to(dev, non_blocking=True)
        ar = ar.to(dev, non_blocking=True); pos = pos.to(dev, non_blocking=True)
        hi = hi.to(dev, non_blocking=True); isaux = isaux.to(dev, non_blocking=True)
        target_mask=target_mask.to(dev,non_blocking=True)
        pl = torch.full((img.shape[0],), plat, dtype=torch.long, device=dev)
        mu = net(img, W, pos, pl, aa, ar, hi)
        core = net.module if ddp else net
        wg = torch.where(isaux > 0, torch.as_tensor(a.lam_aux, device=dev), torch.ones((), device=dev))
        nll,mse,n_targets=supervised_sums(y,mu,core.head.log_theta[hi],wg,target_mask,nb_nll)
        supervised+=n_targets
        count=torch.tensor(float(n_targets),device=dev)
        if ddp:dist.all_reduce(count)
        scale=world/count.clamp_min(1)
        nll=nll*scale;mse=mse*scale
        loss = nll + a.lam_mse * mse
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if coverage is not None and step in coverage.step_ends:
            seen=torch.tensor(supervised,device=dev,dtype=torch.long)
            if ddp:dist.all_reduce(seen)
            epoch=coverage.step_ends.index(step)+1
            if seen.item()!=epoch*coverage.cells:
                raise ValueError(f'Epoch {epoch}: target coverage mismatch {seen.item()}')
            log(f'[coverage] epoch {epoch}/{a.epochs}: global supervised targets={seen.item():,}; '
                f'this epoch={coverage.cells:,}; context excluded from loss')
        if step % 200 == 0:
            log(f"  step {step:>6d}/{total_steps} loss {loss.detach().item():.4f} ({time.time()-t0:.0f}s)")
        if a.no_validation and (step % a.eval_every == 0 or step == total_steps):
            counts=[supervised]
            if ddp:
                counts=[None]*world;dist.all_gather_object(counts,supervised)
            rng=capture_rng_per_rank()
            if rank == 0:
                atomic_torch_save({'lora':{n:p.detach().cpu() for n,p in enc.named_parameters() if p.requires_grad},
                            'inp':core.inp.state_dict(),'se2':core.se2.state_dict(),'head':core.head.state_dict(),
                            'opt':opt.state_dict(),'args':vars(a),'step':step,'world_size':world,
                            'run_contract':run_contract,'gene_head_sha256':gene_hash(space['genes'].tolist()),
                            'stage1_identity':stage1_identity,'eval':None,
                            'coverage_revision':COVERAGE_REVISION if coverage is not None else None,
                            'supervised_cells_per_rank':counts,'rng_per_rank':rng},f'{CKDIR}/{a.tag}_last.pt')
            if ddp: dist.barrier()
        if not a.no_validation and (step % a.eval_every == 0 or step == total_steps):
            if rank == 0:
                agg, per = evaluate(net, va, dev, a.n_anchor, a.eval_workers)
                b = agg["all"]
                log(f"  [eval] step {step:>6d} primary {b['pearson_primary']:.4f} aux {b['pearson_aux']:.4f} "
                    f"hvg50 {b['pearson_hvg50']:.4f}"
                    + "".join(f" | {k} {v['pearson_primary']:.4f}" for k, v in agg.items() if k != "all"))
                hist.append({"step": step, "agg": agg, "per_object": per, "train_loss": loss.detach().item()})
                json.dump({"args": vars(a), "history": hist},
                          open(f"{CKDIR}/{a.tag}_history.json", "w"), indent=2, default=float)
                sd = {"lora": {n_: p.detach().cpu() for n_, p in enc.named_parameters() if p.requires_grad},
                      "inp": core.inp.state_dict(), "se2": core.se2.state_dict(),
                      "head": core.head.state_dict(), "opt": opt.state_dict(),
                      "args": vars(a), "step": step, "eval": agg}
                torch.save(sd, f"{CKDIR}/{a.tag}_last.pt")
                if b["pearson_primary"] > best:
                    best, best_step = b["pearson_primary"], step
                    torch.save(sd, f"{CKDIR}/{a.tag}_best.pt")
                    log(f"  [best] primary {best:.4f} @ step {step}")
            if ddp:
                dist.barrier()
    log(f"[done] {a.tag} ({time.time()-t0:.0f}s)  best primary {best:.4f} @ step {best_step}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
