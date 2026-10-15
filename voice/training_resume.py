"""Checkpoint/cursor checks shared by both fixed-epoch training stages."""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from voice.training_contract import check_checkpoint, load_lora_checked

STAGE1_REVISION = 'stage1_cursor_v1'
STAGE1_RECIPE = ['optimizer', 'target', 'platforms', 'loss', 'adjust_on', 'epochs',
    'max_cells', 'eff_batch', 'chunk', 'lr_lora', 'lr_tower', 'wd', 'warmup', 'nblocks',
    'r', 'alpha', 'dropout', 'seed', 'd4_aug', 'd4_cycle', 'retain_tail', 'no_validation', 'uni2h']
STAGE2_RECIPE = ['gene_emb', 'free_head', 'free_init', 'n_layers', 'd_model', 'span', 'loop_r',
    'nblocks', 'lora_r', 'lora_alpha', 'lora_dropout', 'freeze_frac', 'frozen_enc',
    'lr_lora', 'lr_se2', 'd4', 'n_nb', 'epochs', 'seed', 'no_validation', 'sampling',
    'target_cells', 'context_cells', 'grad_ckpt', 'max_cells', 'max_dist', 'lam_aux', 'lam_mse']


@dataclass(frozen=True)
class Stage1Cursor:
    cells: int
    world: int
    chunk: int
    accumulation: int
    epochs: int
    step: int

    def __post_init__(self):
        if min(self.cells, self.world, self.chunk, self.accumulation, self.epochs) < 1:
            raise ValueError('Positive Stage-1 cursor dimensions required')
        if not 0 <= self.step <= self.total_steps:
            raise ValueError('Resume checkpoint exceeds the fixed epoch budget')

    @property
    def samples_per_rank(self):
        return (self.cells+self.world-1)//self.world

    @property
    def batches_per_epoch(self):
        return (self.samples_per_rank+self.chunk-1)//self.chunk

    @property
    def steps_per_epoch(self):
        return (self.batches_per_epoch+self.accumulation-1)//self.accumulation

    @property
    def total_steps(self):
        return self.steps_per_epoch*self.epochs

    @property
    def epoch(self):
        return self.step//self.steps_per_epoch

    @property
    def skip_batches(self):
        return (self.step%self.steps_per_epoch)*self.accumulation

    @property
    def skip_slots(self):
        return min(self.skip_batches*self.chunk, self.samples_per_rank)

    def valid_per_epoch(self, rank):
        if not 0 <= rank < self.world:
            raise ValueError('Invalid resume rank')
        return max(0, (self.cells-rank+self.world-1)//self.world)

    def supervised_in_epoch(self, rank):
        return min(self.skip_slots, self.valid_per_epoch(rank))

    def supervised_total(self, rank):
        return self.epoch*self.valid_per_epoch(rank)+self.supervised_in_epoch(rank)

    def record(self):
        return {**asdict(self), 'epoch': self.epoch, 'skip_batches': self.skip_batches,
                'skip_slots': self.skip_slots, 'steps_per_epoch': self.steps_per_epoch,
                'supervised_per_rank': [self.supervised_total(r) for r in range(self.world)]}


def stage1_data_identity(slides, paths, cache):
    """Check concat order and completed cache signatures, without reading all PNGs."""
    identity = {'cells': len(paths), 'objects': [name for name, _ in slides]}
    if cache:
        records = []
        for name, _ in slides:
            directory = Path(cache)/'stage1'/name
            records.append({'object': name,
                            'metadata': json.loads((directory/'metadata.json').read_text()),
                            'cells': len(np.load(directory/'paths.npy', mmap_mode='r'))})
        identity['cache_sha256'] = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return identity


def check_recipe(ck, args, keys):
    current = vars(args) if not isinstance(args, dict) else args
    for key in keys:
        if ck['args'].get(key) != current.get(key):
            raise ValueError(f'Resume training recipe differs: {key}')


def check_stage1_resume(ck, args, contract, world, cells, data_identity, init_hash):
    if not args.retain_tail or not args.no_validation or not args.epochs:
        raise ValueError('Stage-1 resume requires retained tails, fixed epochs and no validation')
    check_checkpoint(ck, contract)
    check_recipe(ck, args, STAGE1_RECIPE)
    if ck.get('world_size') != world:
        raise ValueError('Stage-1 resume requires the original DDP world size')
    if any(k not in ck for k in ['lora', 'he_tower', 'scf_tower', 'opt', 'step', 'init_hash']):
        raise ValueError('Incomplete Stage-1 weights/optimizer checkpoint')
    if ck['init_hash'] != init_hash:
        raise ValueError('Stage-1 initialization differs from the saved run')
    if data_identity['cells'] != cells:
        raise ValueError('Stage-1 data identity has an inconsistent effective cell count')
    if ck.get('data_identity', data_identity) != data_identity:
        raise ValueError('Stage-1 data order/cache identity changed')
    cursor = Stage1Cursor(cells, world, args.chunk, max(1,args.eff_batch//args.chunk), args.epochs, int(ck['step']))
    if ck['args']['steps'] != cursor.total_steps or args.steps != cursor.total_steps:
        raise ValueError('Stage-1 effective cell count or total epoch budget changed')
    if 'coverage_cursor' in ck and ck['coverage_cursor'] != cursor.record():
        raise ValueError('Saved Stage-1 coverage cursor is inconsistent')
    return cursor


def restore_stage1_weights(ck, enc, he_tower, target_tower, optimizer, lossfn):
    load_lora_checked(enc, ck['lora'], complete=True)
    he_tower.load_state_dict(ck['he_tower'], strict=True)
    target_tower.load_state_dict(ck['scf_tower'], strict=True)
    if 'lossfn' in ck:
        lossfn.load_state_dict(ck['lossfn'], strict=True)
    elif any(True for _ in lossfn.parameters()):
        raise ValueError('Checkpoint lacks learnable contrastive-loss parameters')
    optimizer.load_state_dict(ck['opt'])


def check_stage2_resume(ck, args, contract, genes, stage1_identity, world, coverage):
    check_checkpoint(ck, contract, genes)
    if 'opt' not in ck:
        raise ValueError('A new-run resume must include optimizer state')
    if ck.get('stage1_identity') != stage1_identity or ck.get('world_size') != world:
        raise ValueError('Resume Stage1 identity or DDP world size changed')
    check_recipe(ck, args, STAGE2_RECIPE)
    if not 0 <= int(ck['step']) <= coverage.total_steps:
        raise ValueError('Resume checkpoint exceeds the fixed epoch budget')
    from voice.full_coverage import REVISION
    expected = [coverage.supervised_before(int(ck['step']),r) for r in range(world)]
    if ck.get('coverage_revision') != REVISION or ck.get('supervised_cells_per_rank') != expected:
        raise ValueError('Resume coverage plan/count differs from the checkpoint')
    return int(ck['step'])


def capture_rng_per_rank():
    state = {'python': random.getstate(), 'numpy': np.random.get_state(),
             'torch_cpu': torch.get_rng_state(),
             'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
    if dist.is_initialized():
        gathered = [None]*dist.get_world_size()
        dist.all_gather_object(gathered, state)
        return gathered
    return [state]


def restore_main_rng(ck, rank, world):
    states = ck.get('rng_per_rank')
    if states is None:
        return False
    if len(states) != world:
        raise ValueError('RNG checkpoint world size differs')
    s = states[rank]
    random.setstate(s['python']); np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch_cpu'].cpu())
    if s['torch_cuda'] is not None:
        torch.cuda.set_rng_state(s['torch_cuda'].cpu())
    return True


def atomic_torch_save(state, destination):
    destination = Path(destination)
    tmp = destination.with_name(destination.name+f'.{os.getpid()}.tmp')
    try:
        torch.save(state, tmp)
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            tmp.unlink()
