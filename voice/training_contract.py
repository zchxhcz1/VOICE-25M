"""Training splits and checkpoint consistency checks."""
import hashlib
import json
from pathlib import Path


def read_ids(path):
    ids = [x.strip() for x in Path(path).read_text().splitlines() if x.strip() and not x.startswith('#')]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate samples: {path}')
    return ids


def dataset_contract(dataset):
    root = Path(dataset)
    ids = {k: read_ids(root/'splits'/f'{k}.txt') for k in
           ['training_set','inslide_test_set','cross_slide_test_set','stage2_global_training']}
    tr, it, ct, s2 = (set(ids[k]) for k in ids)
    if not it <= tr or tr & ct or s2 != tr-it:
        raise ValueError('Invalid Stage1/Stage2/test split')
    for key in ['retrieval_references','gate_fit_references']:
        refs=read_ids(root/'splits'/f'{key}.txt')
        if set(refs)!=s2:
            raise ValueError(f'{key} must match Stage-2 training and exclude all test objects')
        ids[key]=refs
    return {'dataset': str(root.resolve()),
            'manifest_sha256': hashlib.sha256((root/'MANIFEST.csv').read_bytes()).hexdigest(),
            **ids}


def check_members(contract, stage, actual):
    key = 'training_set' if stage == 'stage1' else 'stage2_global_training'
    actual = list(actual)
    if len(actual) != len(set(actual)) or set(actual) != set(contract[key]):
        raise ValueError(f'{stage}: samples do not match the approved {key}; missing='
                         f'{sorted(set(contract[key])-set(actual))}, extra={sorted(set(actual)-set(contract[key]))}')


def gene_hash(genes):
    return hashlib.sha256(json.dumps(list(genes)).encode()).hexdigest()


def checkpoint_identity(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return {'sha256': h.hexdigest()}


def stage1_steps(cells, epochs, world, chunk, accumulation):
    if min(cells, epochs, world, chunk, accumulation) < 1:
        raise ValueError('Positive training budget required')
    per_epoch = (cells+world*chunk*accumulation-1)//(world*chunk*accumulation)
    if per_epoch < 1:
        raise ValueError('No full optimizer batch')
    return epochs*per_epoch


def check_checkpoint(ck, contract, genes=None):
    if ck.get('run_contract') != contract:
        raise ValueError('Checkpoint training split differs from the current corpus')
    if genes is not None and ck.get('gene_head_sha256') != gene_hash(genes):
        raise ValueError('Checkpoint gene-head order differs from this run')


def load_lora_checked(enc, tensors, complete=False):
    own = dict(enc.named_parameters())
    expected = {n for n in own if n.endswith(('.A','.B'))}
    actual = set(tensors)
    if not actual or not actual <= expected or (complete and actual != expected):
        raise ValueError(f'LoRA key mismatch: expected {len(expected)}, received {len(actual)}')
    for n, value in tensors.items():
        if value.shape != own[n].shape:
            raise ValueError(f'LoRA tensor shape mismatch: {n}')
    enc.load_state_dict(tensors, strict=False)
