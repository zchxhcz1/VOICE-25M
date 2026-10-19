"""Align images and scGPT cell embeddings with three full-coverage epochs."""
import os
import time
import hashlib
from pathlib import Path
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from voice.training import contrastive as C
from voice.encoder import build_uni2_local
from voice.recipe import stage1_settings
from voice.clip_coverage import TailDataset, ResumeTailSampler
from voice.training_contract import dataset_contract, check_members, stage1_steps
from voice.training_resume import (Stage1Cursor, stage1_data_identity, check_stage1_resume,
    restore_stage1_weights, capture_rng_per_rank, restore_main_rng, atomic_torch_save, STAGE1_REVISION)

def target_path(slide, embedding_root, n_expr=None):
    path = Path(embedding_root)/f'{slide}.npy'
    if not path.is_file():
        raise FileNotFoundError(f'Missing scGPT embeddings: {path}')
    embedding = np.load(path, mmap_mode='r')
    if embedding.ndim != 2 or embedding.shape[1] != 512:
        raise ValueError(f'{slide}: expected 512-dimensional scGPT cell embeddings')
    if n_expr is not None and len(embedding) != n_expr:
        raise ValueError(f'{slide}: scGPT embeddings are not aligned with expression rows')
    return str(path)


def training_slides(dataset, embedding_root, contract):
    from scipy import sparse
    slides = []
    for platform in ['xenium', 'cosmx']:
        for directory in sorted((Path(dataset)/'training_set'/platform).glob('*/*')):
            if not directory.is_dir():
                continue
            target_path(directory.name, embedding_root, sparse.load_npz(directory/'expression.npz').shape[0])
            slides.append((directory.name, str(directory)))
    check_members(contract, 'stage1', [name for name,_ in slides])
    return slides


def build_data(slides, embedding_root, cache_root):
    from voice.training_cache import cached_arrays, file_stamp, JoinedArrays
    arrays = []
    for name,directory in slides:
        target = target_path(name, embedding_root)
        signature = dict(target=file_stamp(target), boundaries=file_stamp(f'{directory}/patch_cell_boundaries.npz'),
            manifest=file_stamp(f'{directory}/manifest.csv.gz'), per=2**62, seed=1, schema=1)
        def build(root):
            paths, masks, rows = C.load_slide_npz(directory)
            embeddings = np.load(target, mmap_mode='r')[rows].astype(np.float16)
            keep = np.flatnonzero(np.abs(embeddings).sum(1) > 0)
            payload = dict(paths=np.asarray([paths[i] for i in keep], dtype=np.bytes_),
                           W=masks[keep], targets=embeddings[keep])
            for key,value in payload.items():
                np.save(root/f'{key}.tmp.npy', value)
            return ['paths','W','targets']
        data = cached_arrays(f'{cache_root}/stage1/{name}', signature, build)
        arrays.append(data)
        print(f'[data] {name}: {len(data["paths"]):,} usable cells', flush=True)
    return tuple(JoinedArrays([data[key] for data in arrays]) for key in ['paths','W','targets'])


class D4Wrap(Dataset):
    """Apply the same random right-angle rotation and reflection to a crop and its mask."""
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, index):
        image, mask, target = self.base[index]
        orientation = torch.randint(0, 8, (1,)).item()
        rotation, flip = orientation % 4, orientation >= 4
        if rotation:
            image = torch.rot90(image, rotation, dims=(1,2))
            mask = torch.rot90(mask, rotation, dims=(0,1))
        if flip:
            image = image.flip(2)
            mask = mask.flip(1)
        return image.contiguous(), mask.contiguous(), target


def gradcache_step_ddp(chunks, model, he_tower, rna_tower, opt, world, clip_norm=1.0):
    """Compute contrastive gradients in chunks and synchronize across ranks."""
    with torch.no_grad(), torch.autocast("cuda", dtype=C.AMP_DTYPE):
        Zi = [C.he_embed(model, he_tower, b[0], b[1]).float() for b in chunks]
        Zg = [rna_tower(b[2].to(C.dev, non_blocking=True)).float() for b in chunks]
    zi = torch.cat(Zi).detach().requires_grad_(True)
    zg = torch.cat(Zg).detach().requires_grad_(True)
    opt.zero_grad(set_to_none=True)
    mask = torch.cat([batch[3] for batch in chunks]).to(C.dev)
    n = mask.sum()
    total = n.clone()
    if world > 1:
        dist.all_reduce(total)
    loss = C.infonce(zi[mask], zg[mask]) if n.item() else (zi.sum()+zg.sum())*0
    loss = loss * (world*n/total.clamp_min(1))
    loss.backward(); gi, gg = zi.grad, zg.grad
    off = 0
    for batch in chunks:
        img,W,rna=batch[:3]
        c = img.shape[0]
        with torch.autocast("cuda", dtype=C.AMP_DTYPE):
            zi_c = C.he_embed(model, he_tower, img, W)
            zg_c = rna_tower(rna.to(C.dev, non_blocking=True))
        torch.autograd.backward([zi_c.float(), zg_c.float()], [gi[off:off + c], gg[off:off + c]])
        off += c
    params = ([p for p in model.parameters() if p.requires_grad]
              + list(he_tower.parameters()) + list(rna_tower.parameters()))
    if world > 1:                                          # ONE all-reduce, after the last chunk
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM); p.grad /= world
    torch.nn.utils.clip_grad_norm_(params, clip_norm); opt.step()
    return float(loss.detach())


def main(paths):
    a = stage1_settings(tag='stage1', uni2h=paths.uni2h, workers=paths.workers,
                        chunk=paths.chunk, save_every=paths.save_every, resume=paths.resume)
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    if world > 1:
        torch.cuda.set_device(local)
        dist.init_process_group('nccl', timeout=timedelta(hours=4), device_id=torch.device(f'cuda:{local}'))
        C.dev = torch.device(f'cuda:{local}')
    torch.manual_seed(1)
    np.random.seed(1)
    torch.cuda.manual_seed_all(1)
    checkpoint_dir = Path(paths.work_dir)/'checkpoints/alignment'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(paths.work_dir)/'cache'
    contract = dataset_contract(paths.dataset)
    slides = training_slides(paths.dataset, paths.cell_embeddings, contract)
    crops, masks, embeddings = build_data(slides, paths.cell_embeddings, cache_root)
    model = build_uni2_local(C.dev, a.uni2h)
    C.inject_lora(model, 12, 16, 32, 0.05)
    model.to(C.dev)
    mean, std = C.rna_stats(embeddings)
    he_tower = C.HeTower().to(C.dev)
    rna_tower = C.RNATower(mean, std).to(C.dev)
    digest = hashlib.sha256()
    for name,value in sorted(list(C.trainable(model).items())
        + [('he.'+k,v) for k,v in he_tower.state_dict().items()]
        + [('scf.'+k,v) for k,v in rna_tower.state_dict().items()]):
        digest.update(name.encode())
        digest.update(value.detach().float().cpu().numpy().tobytes())
    init_hash = digest.hexdigest()[:16]
    opt = torch.optim.AdamW([{'params':[p for p in model.parameters() if p.requires_grad],'lr':1.5e-4},
        {'params':list(he_tower.parameters())+list(rna_tower.parameters()),'lr':1e-3}], weight_decay=0.01)
    for group in opt.param_groups:
        group['base_lr'] = group['lr']
    generator = torch.Generator().manual_seed(1)
    dataset = TailDataset(D4Wrap(C.CLIPDS(crops, masks, embeddings)))
    sampler = ResumeTailSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=False, seed=1)
    loader = DataLoader(dataset, batch_size=a.chunk, sampler=sampler, generator=generator, drop_last=False,
        num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    accumulation = 2048 // a.chunk
    a.steps = stage1_steps(len(dataset), 3, world, a.chunk, accumulation)
    identity = stage1_data_identity(slides, crops, cache_root)
    step, epoch, cursor, resumed = 0, 0, None, None
    if a.resume:
        resumed = torch.load(a.resume, map_location='cpu', weights_only=False)
        cursor = check_stage1_resume(resumed, a, contract, world, len(dataset), identity, init_hash)
        restore_stage1_weights(resumed, model, he_tower, rna_tower, opt)
        step, epoch = resumed['step'], cursor.epoch
        generator.set_state(resumed['loader_generator_state'].cpu())
    def saved_state(rng):
        return dict(lora=C.trainable(model), he_tower=he_tower.state_dict(), scf_tower=rna_tower.state_dict(),
            args=vars(a), init_hash=init_hash, step=step, world_size=world, run_contract=contract,
            opt=opt.state_dict(), resume_revision=STAGE1_REVISION, data_identity=identity,
            coverage_cursor=Stage1Cursor(len(dataset),world,a.chunk,accumulation,3,step).record(),
            rng_per_rank=rng, loader_generator_state=generator.get_state(), worker_rng_captured=False)
    started = time.time()
    while step < a.steps:
        sampler.set_epoch(epoch)
        skip = cursor.skip_slots if cursor is not None and epoch == cursor.epoch else 0
        sampler.set_start(skip)
        seen = cursor.supervised_in_epoch(rank) if skip else 0
        iterator = iter(loader)
        if resumed is not None:
            restore_main_rng(resumed, rank, world)
            resumed = None
        chunks = []
        for index,batch in enumerate(iterator):
            seen += int(batch[3].sum())
            chunks.append(batch)
            if len(chunks) < accumulation and index != len(loader)-1:
                continue
            for group in opt.param_groups:
                group['lr'] = group['base_lr'] * min(1.0, (step+1)/150)
            loss = gradcache_step_ddp(chunks, model, he_tower, rna_tower, opt, world)
            chunks = []
            step += 1
            if step % 25 == 0:
                print(f'[train] step {step}/{a.steps}; loss={loss:.4f}; {time.time()-started:.0f}s', flush=True)
            if step % a.save_every == 0 or step == a.steps:
                rng = capture_rng_per_rank()
                if rank == 0:
                    atomic_torch_save(saved_state(rng), f'{checkpoint_dir}/stage1_step{step}.pt')
            if step >= a.steps:
                break
        count = torch.tensor(seen, device=C.dev, dtype=torch.long)
        if world > 1: dist.all_reduce(count)
        if count.item() != len(dataset):
            raise ValueError('Image-RNA training did not cover every usable cell exactly once')
        if rank == 0:
            print(f'[coverage] epoch {epoch+1}/3: {count.item():,} supervised cells', flush=True)
        epoch += 1
        cursor = None
    if resumed is not None:
        restore_main_rng(resumed, rank, world)
    rng = capture_rng_per_rank()
    if rank == 0:
        atomic_torch_save(saved_state(rng), f'{checkpoint_dir}/stage1_final.pt')
        print(f'[done] {checkpoint_dir}/stage1_final.pt', flush=True)
    if world > 1: dist.destroy_process_group()
