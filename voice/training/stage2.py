"""Train expression prediction for three epochs with complete target-cell coverage."""
import os
import json
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
from voice.encoder import build_uni2_local
from voice.training import contrastive as C
from voice.training.cell_inputs import cells_of
from voice.training_contract import (dataset_contract, check_members, check_checkpoint,
    gene_hash, checkpoint_identity, load_lora_checked)
from voice.training_resume import (check_stage2_resume, capture_rng_per_rank, restore_main_rng, atomic_torch_save)
from voice.tissue import object_directory
from voice.full_coverage import CoveragePlan, FullCoverageDataset, supervised_sums, REVISION
from voice.head import DirectHead, nb_nll, cell_areas
from voice.se2 import SpatialTransformer
from voice.recipe import stage2_settings

class LiveStore:
    """All effective cells of one slide, with images, masks, counts and micron positions."""

    def __init__(self, obj, d, plat, space, feature_root, cache):
        cm = f"{feature_root}/{'cosmx' if plat else 'xenium'}/{obj}/cmask.npz"
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
        if cache:
            from voice.training_cache import cached_arrays, file_stamp, array_stamp
            signature={'boundaries':file_stamp(f'{d}/patch_cell_boundaries.npz'),
                       'mask':file_stamp(cm) if os.path.exists(cm) else None,
                       'rows':array_stamp(er),'max_cells':0,'schema':1}
            def build_masks(root):
                np.save(root/'W.tmp.npy',W);return ['W']
            W=cached_arrays(f'{cache}/stage2/{obj}/masks',signature,build_masks)['W']
        self.paths, self.W, self.pos = np.array(paths, dtype=np.bytes_), W, pos

        X = sp.load_npz(f"{d}/expression.npz")
        syms = pd.read_csv(f"{d}/genes.tsv", sep="\t").iloc[:, 0].astype(str).tolist()
        head = {g: i for i, g in enumerate(space["genes"].tolist())}
        cols, hidx, seen = [], [], set()
        for j, g in enumerate(syms):
            if g in head and g not in seen:
                seen.add(g); cols.append(j); hidx.append(head[g])
        self.head_idx = np.asarray(hidx, np.int64)
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
        self.obj, self.n = obj, len(er)

    def load(self, idx):
        """Decode cell crops without expression-stage augmentation."""
        img = np.stack([np.asarray(Image.open(self.paths[i].decode()).convert("RGB").resize((224, 224), Image.BILINEAR),
                                   np.uint8).transpose(2, 0, 1) for i in idx])
        W = self.W[idx]
        return img, W


class ExpressionModel(nn.Module):
    def __init__(self, enc, gene_emb):
        super().__init__()
        self.enc = enc
        self.inp = nn.Linear(1536, 256)
        self.se2 = SpatialTransformer()
        self.head = DirectHead(gene_emb)

    def forward(self, img, W, pos, aa, ar, hi):
        with torch.autocast('cuda', dtype=C.AMP_DTYPE):
            h = C.pooled_feat(self.enc, C.norm(img), W)
        return self.head(self.se2(self.inp(h.float()), pos), aa, ar, hi)


def one(batch):
    return batch[0]


def freeze_first_lora(enc):
    """Freeze the first three of the last twelve adapted blocks."""
    start = len(enc.blocks)-12
    for block in enc.blocks[start:start+3]:
        for module in [block.attn.qkv, block.attn.proj]:
            module.A.requires_grad_(False)
            module.B.requires_grad_(False)


def main(paths):
    a = stage2_settings(tag='stage2', lora_ckpt=paths.stage1, workers=paths.workers,
                        save_every=paths.save_every, resume=paths.resume)
    root = Path(paths.work_dir)
    checkpoint_dir = root/'checkpoints/expression'
    feature_root, cache = root/'features', root/'cache'
    contract = dataset_contract(paths.dataset)
    names = contract['stage2_global_training']
    a.train_objects = ','.join(names)
    ddp = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if ddp:
        dist.init_process_group('nccl')
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    else:
        rank, world = 0, 1
    dev = torch.device('cuda')
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    log = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    space = np.load(root/'direct/gene_space_all.npz', allow_pickle=False)
    lookup = {o:int(p) for o,p in zip(space['objects'].tolist(),space['platform'])}
    stores = []
    for name in names:
        directory = object_directory(paths.dataset, name)
        store = LiveStore(name, str(directory), lookup[name], space, feature_root, cache)
        stores.append(store)
        log(f'[data] {name}: {store.n:,} cells; {len(store.head_idx):,} measured genes')
    check_members(contract, 'stage2', [s.obj for s in stores])
    coverage = CoveragePlan(stores, a.epochs, a.target_cells, a.seed, world, cache)
    a.steps = coverage.blocks
    log(f'[coverage] {coverage.cells:,} targets/epoch; {coverage.total_steps:,} steps/rank; '
        f'{a.target_cells} targets + {a.context_cells} context cells/block')
    enc = build_uni2_local(dev, paths.uni2h)
    C.inject_lora(enc, a.nblocks, a.lora_r, a.lora_alpha, a.lora_dropout)
    ck = torch.load(a.lora_ckpt, map_location='cpu', weights_only=False)
    check_checkpoint(ck, contract)
    if ck['args']['target'] != 'scgpt' or ck['args']['epochs'] != 3 or ck['step'] != ck['args']['steps']:
        raise ValueError('Initialize expression training with a completed three-epoch scGPT Stage-1 checkpoint')
    load_lora_checked(enc, ck['lora'], complete=True)
    identity = checkpoint_identity(a.lora_ckpt)
    for name, param in enc.named_parameters():
        param.requires_grad_(name.endswith(('.A', '.B')))
    freeze_first_lora(enc)
    enc.set_grad_checkpointing(True)
    table = np.load(root/'direct/gene_emb_scgpt.npz', allow_pickle=False)
    if not np.array_equal(table['genes'], space['genes']):
        raise ValueError('scGPT gene embeddings and output gene order differ')
    model = ExpressionModel(enc, table['emb'].astype(np.float32)).to(dev)
    resumed = None
    if a.resume:
        resumed = torch.load(a.resume, map_location='cpu', weights_only=False)
        check_stage2_resume(resumed, a, contract, space['genes'].tolist(), identity, world, coverage)
        if set(resumed['lora']) != {n for n,p in enc.named_parameters() if p.requires_grad}:
            raise ValueError('Resume checkpoint has different trainable image adapters')
        load_lora_checked(enc, resumed['lora'])
        for key in ['inp', 'se2', 'head']:
            getattr(model, key).load_state_dict(resumed[key], strict=True)
    lora = [p for p in enc.parameters() if p.requires_grad]
    rest = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('enc.')]
    opt = torch.optim.AdamW([{'params':lora,'lr':3e-5}, {'params':rest,'lr':1e-4}], weight_decay=0.0)
    if resumed is not None:
        opt.load_state_dict(resumed['opt'])
    net = nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ['LOCAL_RANK'])],
        gradient_as_bucket_view=True, broadcast_buffers=False) if ddp else model
    step = int(resumed['step']) if resumed is not None else 0
    if step == coverage.total_steps:
        log('[done] Three training epochs are already complete')
        if ddp: dist.destroy_process_group()
        return
    supervised = coverage.supervised_before(step, rank)
    dataset = FullCoverageDataset(coverage, rank, a.context_cells, a.max_dist, start=step)
    loader = DataLoader(dataset, batch_size=1, num_workers=a.workers, collate_fn=one,
        pin_memory=True, persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
    iterator = iter(loader)
    if resumed is not None:
        restore_main_rng(resumed, rank, world)
    started = time.time()
    for img, W, y, aa, ar, pos, hi, mask in iterator:
        step += 1
        img, W, y, aa, ar, pos, hi, mask = [v.to(dev, non_blocking=True) for v in (img,W,y,aa,ar,pos,hi,mask)]
        mu = net(img, W, pos, aa, ar, hi)
        weights = torch.ones(len(hi), device=dev)
        nll, mse, count = supervised_sums(y, mu, model.head.log_theta[hi], weights, mask, nb_nll)
        supervised += count
        total = torch.tensor(float(count), device=dev)
        if ddp: dist.all_reduce(total)
        scale = world / total.clamp_min(1)
        loss = nll * scale + mse * scale
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step in coverage.step_ends:
            seen = torch.tensor(supervised, device=dev, dtype=torch.long)
            if ddp: dist.all_reduce(seen)
            epoch = coverage.step_ends.index(step) + 1
            if seen.item() != epoch * coverage.cells:
                raise ValueError('Target-cell coverage differs from the epoch budget')
            log(f'[coverage] epoch {epoch}/3: {seen.item():,} supervised targets')
        if step % 200 == 0:
            log(f'[train] step {step}/{coverage.total_steps}; loss={loss.detach().item():.4f}; {time.time()-started:.0f}s')
        if step % a.save_every == 0 or step == coverage.total_steps:
            counts = [supervised]
            if ddp:
                counts = [None]*world
                dist.all_gather_object(counts, supervised)
            rng = capture_rng_per_rank()
            if rank == 0:
                atomic_torch_save(dict(lora={n:p.detach().cpu() for n,p in enc.named_parameters() if p.requires_grad},
                    inp=model.inp.state_dict(), se2=model.se2.state_dict(), head=model.head.state_dict(),
                    opt=opt.state_dict(), args=vars(a), step=step, world_size=world, run_contract=contract,
                    gene_head_sha256=gene_hash(space['genes'].tolist()), stage1_identity=identity,
                    coverage_revision=REVISION, supervised_cells_per_rank=counts, rng_per_rank=rng),
                    f'{checkpoint_dir}/stage2_last.pt')
            if ddp: dist.barrier()
    if ddp: dist.destroy_process_group()
    log(f'[done] {checkpoint_dir}/stage2_last.pt')
