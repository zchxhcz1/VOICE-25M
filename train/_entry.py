"""Path and hardware arguments shared by the two fixed training stages."""
from pathlib import Path
import argparse


def arguments(stage, argv=None):
    p = argparse.ArgumentParser(description=f'Train VOICE-25M stage {stage} for three epochs')
    p.add_argument('--dataset', required=True, help='Prepared paired corpus; see docs/training.md')
    p.add_argument('--work-dir', required=True, help='Output directory for checkpoints and caches')
    p.add_argument('--uni2h', required=True, help='Local UNI2-h weights directory')
    if stage == 1:
        p.add_argument('--cell-embeddings', required=True, help='Row-aligned <sample>.npy scGPT embeddings')
        p.add_argument('--chunk', type=int, default=256, help='Encoder chunk size; must divide 2048')
    else:
        p.add_argument('--stage1', required=True, help='Completed Stage-1 checkpoint')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--save-every', type=int, default=500 if stage == 1 else 5000)
    p.add_argument('--resume', default='', help='Restore a complete checkpoint from this training stage')
    a = p.parse_args(argv)
    if a.workers < 0 or a.save_every < 1:
        p.error('workers must be nonnegative and save-every must be positive')
    if stage == 1 and (a.chunk < 1 or 2048 % a.chunk):
        p.error('chunk must be a positive divisor of 2048')
    for key in ['dataset', 'work_dir', 'uni2h'] + (['cell_embeddings'] if stage == 1 else ['stage1']):
        setattr(a, key, str(Path(getattr(a, key)).expanduser().resolve()))
    Path(a.work_dir).mkdir(parents=True, exist_ok=True)
    return a
