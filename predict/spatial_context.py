"""Training-matched target ownership and halo context for inference.

No expression values are read. Reuse the actual training partition/halo functions;
only the target prefix is committed to slide outputs, exactly once per cell.
"""
import zlib
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
from voice.full_coverage import partition_cells, halo_cells, REVISION
from voice.recipe import TARGET_CELLS, CONTEXT_CELLS, RADIUS_UM, SEED


class TargetContextPlan:
    def __init__(self, pos, object_name):
        self.pos = np.asarray(pos)
        if self.pos.ndim != 2 or self.pos.shape[1] != 2 or not len(self.pos):
            raise ValueError('Expected nonempty [cells, 2] micron positions')
        self.target_cells = TARGET_CELLS
        self.context_cells = CONTEXT_CELLS
        self.radius = RADIUS_UM
        self.seed = SEED
        self.object_name, self.epoch = str(object_name), 0
        self.object_seed = (SEED + zlib.crc32(self.object_name.encode())) % (2**31 - 1)
        self.order, self.ptr = partition_cells(self.pos, self.target_cells, self.object_seed, self.epoch)
        self.tree = None

    def __len__(self):
        return len(self.ptr) - 1

    def targets(self, block):
        return self.order[self.ptr[block]:self.ptr[block + 1]]

    def indices(self, block):
        targets = self.targets(block)
        if self.tree is None:
            self.tree = cKDTree(self.pos)
        context = halo_cells(self.pos, targets, self.tree, self.context_cells, self.radius)
        return np.r_[targets, context], len(targets)

    def metadata(self):
        return dict(mode='target_context', revision=REVISION, max_target_cells=self.target_cells,
                    max_context_cells=self.context_cells, context_radius_um=self.radius,
                    seed=self.seed, partition_object=self.object_name, partition_epoch=self.epoch,
                    object_seed=int(self.object_seed), n_blocks=len(self),
                    output_policy='targets_once_only', partition_fixed_for_inference=True)


class TargetContextDataset(Dataset):
    def __init__(self, source, plan):
        self.source, self.plan = source, plan

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, block):
        indices, n_targets = self.plan.indices(block)
        images, masks = self.source.batch(indices)
        return (torch.from_numpy(images), torch.from_numpy(masks),
                torch.from_numpy(indices), n_targets)


def commit_target_rows(indices, n_targets, counts, payloads):
    """Scatter each owned prediction once; context rows never overwrite outputs."""
    indices = np.asarray(indices)
    if not 0 < n_targets <= len(indices):
        raise ValueError('Invalid owned target prefix')
    owned = indices[:n_targets]
    if len(np.unique(owned)) != n_targets or np.any(counts[owned] != 0):
        raise ValueError('A target cell would be written more than once')
    order = np.argsort(owned)
    for destination, values in payloads:
        if len(values) != len(indices):
            raise ValueError('Model output does not cover the complete target/context batch')
        destination[owned[order]] = values[:n_targets][order]
    counts[owned] += 1
    return n_targets


def prediction_protocol():
    """Settings shared by reference calibration and deployed prediction."""
    return dict(mode='target_context', revision=REVISION, max_target_cells=TARGET_CELLS,
                max_context_cells=CONTEXT_CELLS, context_radius_um=RADIUS_UM, seed=SEED,
                partition_epoch=0, output_policy='targets_once_only')


def check_prediction_protocol(metadata):
    if any(metadata.get(key) != value for key,value in prediction_protocol().items()):
        raise ValueError('Fusion calibration and spatial prediction settings differ')
