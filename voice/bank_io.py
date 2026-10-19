"""Reference bank I/O and consistency checks."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import os
import numpy as np

SCHEMA = 'voice25m-reference-bank'
FEATURE_PROTOCOL = 'masked_cell_features'
VALUE_PROTOCOL = 'standardized_log_expression'
ARTIFACTS = ('keys', 'values', 'panel', 'counts', 'labels', 'PCA_mean',
             'PCA_components', 'PCA_training_rows')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def close_array(array):
    # h5py datasets own a file handle; numpy memmaps own a read-only mapping.
    if hasattr(array, 'file'):
        array.file.close()
    elif isinstance(array, np.memmap):
        array._mmap.close()


@contextmanager
def exclusive(path):
    """Kernel lock releases on exit, including a killed process; no stale PID locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another writer owns {path}') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_feature(meta_path, model):
    meta = json.loads(Path(meta_path).read_text())
    for key in ('checkpoint_sha256', 'feature_protocol_id'):
        require(meta[key] == model[key], f'{meta_path}: {key} mismatch')
    require(meta['dim'] == 1536, f'{meta_path}: expected 1536-dimensional Phase-1')
    require(meta['output_field'] == 'phase1',
            f'{meta_path}: expected raw Phase-1, not Stage-2 E or Z')
    return meta
