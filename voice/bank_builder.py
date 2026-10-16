"""Construct a reference bank by clustering cell features within each reference slide."""
from pathlib import Path
import argparse
import gc
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from voice.retrieval import read_array, sha, signature, original_keys
from voice.bank_io import (ARTIFACTS, SCHEMA, close_array, exclusive, fingerprint,
                      require, validate_feature, write_json)


def choose_backend(name, device):
    if name == 'torch':
        return name
    try:
        import faiss
        import faiss.contrib.torch_utils
        require(device.type == 'cuda' and hasattr(faiss, 'StandardGpuResources'),
                'Faiss GPU backend is unavailable')
        return 'faiss'
    except (ImportError, ValueError):
        if name == 'faiss':
            raise RuntimeError('Install Faiss GPU or use --backend torch')
        return 'torch'


@torch.no_grad()
def assign_torch(X, centers, labels, row_chunk=8192, center_chunk=4096):
    # Exhaustive float32 L2; chunking bounds memory, without ANN or subsampling.
    norms = (centers * centers).sum(1)
    for begin in range(0, len(X), row_chunk):
        block = X[begin:begin + row_chunk]
        xn = (block * block).sum(1, keepdim=True)
        best = torch.full((len(block),), float('inf'), device=X.device)
        ids = torch.zeros(len(block), dtype=torch.long, device=X.device)
        for c0 in range(0, len(centers), center_chunk):
            c1 = min(c0 + center_chunk, len(centers))
            distance = xn + norms[c0:c1][None, :] - 2 * (block @ centers[c0:c1].T)
            value, idx = distance.min(1)
            improved = value < best  # first centre wins exact ties
            best[improved], ids[improved] = value[improved], idx[improved] + c0
        labels[begin:begin + len(block)] = ids


@torch.no_grad()
def summarize(E, Y, labels, nclusters, device):
    counts = torch.bincount(labels, minlength=nclusters)
    key_sum = torch.zeros((nclusters, 1536), device=device, dtype=torch.float32)
    value_sum = torch.zeros((nclusters, Y.shape[1]), device=device, dtype=torch.float32)
    for begin in range(0, len(E), 8192):
        end = min(begin + 8192, len(E))
        key_sum.index_add_(0, labels[begin:end], original_keys(E[begin:end], device).float())
        values = np.array(Y[begin:end], dtype=np.float32, copy=True)
        require(np.isfinite(values).all(), f'Nonfinite expression at rows {begin}:{end}')
        value_sum.index_add_(0, labels[begin:end], torch.as_tensor(values, device=device))
    keys = F.normalize(key_sum / counts[:, None], dim=1).half().cpu().numpy()
    values = (value_sum / counts[:, None]).cpu().numpy()
    require(np.isfinite(keys).all() and np.isfinite(values).all(), 'Nonfinite summary')
    return keys, values, counts.cpu().numpy().astype(np.int64)


def validate_completed(path, fingerprint_value, obj):
    meta = json.loads((path / 'complete.json').read_text())
    require(meta['cache_fingerprint'] == fingerprint_value and meta['object'] == obj,
            f'{path}: incompatible completed object')
    for name in ARTIFACTS:
        require(sha(path / f'{name}.npy') == meta[f'{name}_sha256'], f'{path}: corrupt {name}')
    return meta


@torch.no_grad()
def build(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    backend = choose_backend(args.backend, device)
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    objects = manifest['objects']
    expected = args.expected_objects or len(objects)
    require(len(objects) == expected and 0 < expected <= 256,
            f'Expected all {expected} reference objects, got {len(objects)}')
    require(len({x['object'] for x in objects}) == len(objects), 'Duplicate objects')
    checkpoint_sha = sha(manifest['checkpoint'])
    gene_sha = sha(manifest['gene_space'])
    with np.load(manifest['gene_space'], allow_pickle=False) as space:
        genes = space['genes']
    require(len(set(genes.tolist())) == len(genes), 'Duplicate global genes')
    identity = dict(schema=SCHEMA, checkpoint_sha256=checkpoint_sha, gene_space_sha256=gene_sha,
                    feature_protocol_id=manifest['feature_protocol_id'],
                    value_protocol_id=manifest['value_protocol_id'],
                    manifest_sha256=sha(manifest_path), target_size=100, PCA_dimensions=64,
                    max_PCA_rows=100000, kmeans_iterations=10, per_object_seed='object_index',
                    expected_objects=expected, objects=[o['object'] for o in objects],
                    retrieval_K=50, tau=0.03, backend=backend,
                    torch_version=str(torch.__version__), build_device=str(device),
                    n_genes=len(genes))
    if backend == 'faiss':
        import faiss
        identity['faiss_version'] = faiss.__version__
        faiss.omp_set_num_threads(4)
    source_signatures = []
    for obj in objects:
        require(obj['split'] == 'train', f'{obj["object"]}: expected training reference')
        validate_feature(obj['feature_meta'], identity, obj['E'])
        source_signatures.append(dict(arrays={k: signature(obj[k]) for k in ('E', 'Y', 'panel')},
                                      feature_meta_sha256=sha(obj['feature_meta'])))
    identity['source_signatures'] = source_signatures
    cache_fingerprint = fingerprint(identity)
    model = dict(identity, cache_fingerprint=cache_fingerprint)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with exclusive(root / '.build.lock'):
        if (root / 'model.json').exists():
            require(json.loads((root / 'model.json').read_text()) == model,
                    'Bank input/configuration changed; choose a new output directory')
        else:
            write_json(root / 'model.json', model)
        stats = []
        for si, obj in enumerate(objects):
            started = time.perf_counter()
            path = root / 'summaries' / f'object_{si:02d}'
            if (path / 'complete.json').exists():
                stats.append(validate_completed(path, cache_fingerprint, obj['object']))
                print('reuse completed', obj['object'], flush=True)
                continue
            path.mkdir(parents=True, exist_ok=True)
            E, Y, panel_source = [read_array(obj[k]) for k in ('E', 'Y', 'panel')]
            try:
                panel = np.asarray(panel_source[:], dtype=np.int64)
                n = len(E)
                require(E.shape == (n, 1536) and Y.shape == (n, len(panel)), 'E/Y/panel shape mismatch')
                require(len(panel) > 0 and len(np.unique(panel)) == len(panel), 'Invalid panel')
                require(np.all((panel >= 0) & (panel < len(genes))), 'Panel outside global genes')
                require(min(n, 100000) >= 64, f'{obj["object"]}: PCA64 needs >=64 cells')
                if 'n_cells' in obj:
                    require(n == obj['n_cells'], f'{obj["object"]}: missing full-cell rows')
                torch.manual_seed(si)
                generator = torch.Generator(device='cpu').manual_seed(si)
                rows = np.sort(torch.randperm(n, generator=generator)[:min(n, 100000)].numpy())
                sample = original_keys(E[rows], device).float()
                mean = sample.mean(0)
                _, _, components = torch.pca_lowrank(sample - mean, q=64, center=False)
                np.save(path / 'PCA_mean.npy', mean.cpu().numpy())
                np.save(path / 'PCA_components.npy', components.cpu().numpy())
                np.save(path / 'PCA_training_rows.npy', rows)
                del sample
                X = torch.empty((n, 64), device=device, dtype=torch.float32)
                for begin in range(0, n, 32768):
                    X[begin:begin + 32768] = (original_keys(E[begin:begin + 32768], device).float() - mean) @ components
                del mean, components
                generator = torch.Generator(device='cpu').manual_seed(si)
                nclusters = max(1, n // 100)
                centers = X[torch.randperm(n, generator=generator)[:nclusters].to(device)].clone()
                labels = torch.empty(n, device=device, dtype=torch.long)
                resource = None
                if backend == 'faiss':
                    resource = faiss.StandardGpuResources()
                    resource.setTempMemory(256 * 1024**2)
                    # Faiss and Torch share the default CUDA stream (safe tensor interop).
                    resource.setDefaultNullStreamAllDevices()
                for iteration in range(10):
                    if backend == 'faiss':
                        cfg = faiss.GpuIndexFlatConfig()
                        cfg.device = device.index if device.index is not None else torch.cuda.current_device()
                        cfg.useFloat16 = False
                        index = faiss.GpuIndexFlatL2(resource, 64, cfg)
                        index.add(centers)
                        for begin in range(0, n, 8192):
                            _, ids = index.search(X[begin:begin + 8192], 1)
                            labels[begin:begin + 8192] = ids[:, 0]
                        del index
                    else:
                        assign_torch(X, centers, labels)
                    sums = torch.zeros_like(centers)
                    counts = torch.bincount(labels, minlength=nclusters)
                    for begin in range(0, n, 32768):
                        sums.index_add_(0, labels[begin:begin + 32768], X[begin:begin + 32768])
                    valid = counts > 0
                    centers[valid] = sums[valid] / counts[valid, None]
                    del sums, counts, valid
                    print(f'{obj["object"]}: iteration {iteration + 1}/10 full cells={n}', flush=True)
                # Keep the tenth assignment; no eleventh search after the last update.
                _, labels = torch.unique(labels, sorted=True, return_inverse=True)
                nclusters = int(labels.max().item()) + 1
                del X, centers
                keys, values, counts = summarize(E, Y, labels, nclusters, device)
                np.save(path / 'keys.npy', keys)
                np.save(path / 'values.npy', values)
                np.save(path / 'panel.npy', panel)
                np.save(path / 'counts.npy', counts)
                np.save(path / 'labels.npy', labels.cpu().numpy().astype(np.int32))
                require(int(counts.sum()) == n, 'Lost cells during aggregation')
                current = dict(arrays={k: signature(obj[k]) for k in ('E', 'Y', 'panel')},
                               feature_meta_sha256=sha(obj['feature_meta']))
                require(current == source_signatures[si], 'Input changed during build')
                complete = dict(object=obj['object'], object_index=si, size=100, n_cells=n,
                                n_communities=nclusters, cache_fingerprint=cache_fingerprint,
                                checkpoint_sha256=checkpoint_sha, source=current,
                                seconds=time.perf_counter() - started)
                for name in ARTIFACTS:
                    complete[f'{name}_sha256'] = sha(path / f'{name}.npy')
                write_json(path / 'complete.json', complete)
                stats.append(complete)
                print(json.dumps(complete), flush=True)
                del labels, keys, values, counts, resource
            finally:
                for array in (E, Y, panel_source):
                    close_array(array)
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        write_json(root / 'bank_complete.json', dict(cache_fingerprint=cache_fingerprint,
                   n_objects=len(stats), n_cells=sum(s['n_cells'] for s in stats),
                   n_communities=sum(s['n_communities'] for s in stats)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--backend', choices=('auto', 'faiss', 'torch'), default='auto')
    parser.add_argument('--expected-objects', type=int, default=None,
                        help='Expected reference count; default inferred from manifest')
    build(parser.parse_args())


if __name__ == "__main__":
    main()
