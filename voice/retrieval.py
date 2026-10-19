"""Cluster-based expression retrieval with measured-gene coverage and shared tie weights."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import struct
import zipfile
import numpy as np
import torch
import torch.nn.functional as F

from voice.bank_io import ARTIFACTS, SCHEMA, close_array, fingerprint, require

TAU = 0.03

def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()

def signature(spec):
    path = Path(spec['path']).resolve()
    stat = path.stat()
    return dict(path=str(path), key=spec.get('key'), size=stat.st_size,
                mtime_ns=stat.st_mtime_ns)

def read_array(spec):
    path = Path(spec['path'])
    if path.suffix == '.npy':
        return np.load(path, mmap_mode='r', allow_pickle=False)
    if path.suffix == '.npz':
        return npz_matrix(path, spec['key'])
    if path.suffix in ['.h5', '.hdf5']:
        import h5py
        # Dataset keeps the read-only file handle alive while it is in use.
        handle = h5py.File(path, 'r')
        try:
            return handle[spec['key']]
        except Exception:
            handle.close()
            raise
    raise ValueError('Use NPY, uncompressed NPZ, or HDF5')


def npz_matrix(path, member):
    """Read existing uncompressed npz matrices in place, without whole-file copies."""
    with zipfile.ZipFile(path) as z:
        info=z.getinfo(member+'.npy')
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f'Expected existing uncompressed archive: {path}/{member}')
    with open(path,'rb') as f:
        f.seek(info.header_offset)
        fields=struct.unpack('<IHHHHHIIIHH',f.read(30))
        f.seek(fields[-2]+fields[-1],1)
        version=np.lib.format.read_magic(f)
        shape,fortran,dtype=(np.lib.format.read_array_header_1_0(f) if version==(1,0)
                             else np.lib.format.read_array_header_2_0(f))
        offset=f.tell()
    require(not fortran and not dtype.hasobject, 'NPZ requires C-order non-object arrays')
    return np.memmap(path,dtype=dtype,mode='r',offset=offset,shape=shape)

def original_keys(E, device='cuda'):
    data = np.array(E, dtype=np.float32, copy=True, order='C')
    require(data.ndim == 2 and data.shape[1] == 1536 and np.isfinite(data).all(),
            'Expected finite [N,1536] raw Phase-1 features')
    return F.normalize(torch.as_tensor(data, device=device), dim=-1).half()

def groups_for(panel,banks):
    groups={}; bank_sets=[set(b.panel.tolist()) for b in banks]
    for j,g in enumerate(panel):
        hits=tuple(i for i,s in enumerate(bank_sets) if int(g) in s)
        if hits: groups.setdefault(hits,[]).append(j)
    return list(groups.items())

class ReferenceBank:
    def __init__(self, directory, device="cuda", verify_hashes=True, exclude=()):
        self.device = torch.device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        directory = Path(directory)
        self.model = json.loads((directory / 'model.json').read_text())
        require(self.model.get('schema') == SCHEMA, 'Unsupported bank schema')
        identity = {k: v for k, v in self.model.items() if k != 'cache_fingerprint'}
        require(fingerprint(identity) == self.model['cache_fingerprint'], 'Bank model fingerprint mismatch')
        require(self.model['target_size'] == 100, 'Expected a cluster target size of 100 cells')
        require(len(self.model['objects']) == self.model['expected_objects'], 'Incomplete object list')
        marker = directory / 'bank_complete.json'
        require(marker.is_file(), 'Bank has not completed all objects')
        completed = json.loads(marker.read_text())
        require(completed['n_objects'] == self.model['expected_objects'], 'Wrong completed object count')
        require(completed['cache_fingerprint'] == self.model['cache_fingerprint'],
                'Bank completion fingerprint mismatch')
        require(self.model['retrieval_K'] == 50 and self.model['tau'] == TAU, 'Expected 50 neighbors and a temperature of 0.03')
        banks = []
        for si, obj in enumerate(self.model['objects']):
            if obj in set(exclude):
                continue
            path = self._object_path(directory, si)
            panel = np.load(path / 'panel.npy', allow_pickle=False)
            meta = json.loads((path / 'complete.json').read_text())
            require(meta['object'] == obj and meta['size'] == 100, 'Object metadata mismatch')
            for name in ARTIFACTS:
                require((path / f'{name}.npy').is_file(), f'Missing artifact: {path}/{name}')
                if verify_hashes:
                    require(sha(path / f'{name}.npy') == meta[f'{name}_sha256'], f'Corrupt artifact: {path}/{name}')
            require(panel.ndim == 1 and len(np.unique(panel)) == len(panel), 'Invalid bank panel')
            require(np.all((panel >= 0) & (panel < self.model['n_genes'])), 'Bank panel out of range')
            banks.append(SimpleNamespace(obj=obj, panel=panel, file_index=si))
        require(0 < len(banks) <= 256, 'Expected 1..256 objects')  # uint8 source IDs.
        self._load_cache(banks, 100, directory)

    def _object_path(self, directory, index):
        path = directory / 'summaries' / f'object_{index:02d}'
        if path.is_dir():
            return path
        raise FileNotFoundError(f'Missing reference object directory: {path}')

    def _load_cache(self, banks, target, directory):
        # Preallocate the expression pool to bound host and device memory.
        # Avoid holding a second whole expression pool during concatenation.
        self.banks, self.size = banks, target
        self.keys, self.offsets = [], []
        row_starts, sources, values = [], [], []
        for si, bank in enumerate(banks):
            path = self._object_path(directory, bank.file_index)
            meta = json.loads((path/'complete.json').read_text())
            require(meta['object']==bank.obj and meta['cache_fingerprint']==self.model['cache_fingerprint'], 'Stale object cache')
            values.append(np.load(path/'values.npy', mmap_mode='r'))
        self.pool=torch.empty(sum(Y.size for Y in values),device=self.device,dtype=torch.float32)
        cell_offset=value_offset=0
        for si,(bank,Y) in enumerate(zip(banks,values)):
            path=self._object_path(directory, bank.file_index)
            E=torch.as_tensor(np.load(path/'keys.npy',allow_pickle=False),device=self.device)
            require(E.dtype == torch.float16 and E.shape[1:] == (1536,), 'Invalid community keys')
            require(Y.shape==(len(E),len(bank.panel)) and Y.dtype==np.float32, 'Invalid community values')
            self.keys.append(E)
            self.offsets.append(cell_offset)
            row_starts.append(torch.arange(len(Y),device=self.device,dtype=torch.long)*Y.shape[1]+value_offset)
            sources.append(torch.full((len(Y),),si,device=self.device,dtype=torch.uint8))
            self.pool[value_offset:value_offset+Y.size]=torch.as_tensor(np.array(Y.reshape(-1), copy=True),device=self.device)
            cell_offset+=len(Y);value_offset+=Y.size
            close_array(Y)
        self.row_start=torch.cat(row_starts)
        self.source=torch.cat(sources)
        self.n_communities=cell_offset
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def configure(self, panel):
        panel = np.asarray(panel, dtype=np.int64)
        require(panel.ndim == 1 and len(np.unique(panel)) == len(panel), 'Invalid query panel')
        require(np.all((panel >= 0) & (panel < self.model['n_genes'])), 'Query panel outside gene space')
        self.panel = panel
        self.groups = groups_for(panel, self.banks)
        group_of = np.full(len(panel), -1, np.int64)
        self.colmap = np.full((len(self.banks), len(panel)), -1, np.int32)
        for gi, (_, positions) in enumerate(self.groups):
            group_of[positions] = gi
        gene_position = {int(g): i for i, g in enumerate(panel)}
        for si, b in enumerate(self.banks):
            for col, g in enumerate(b.panel):
                if int(g) in gene_position:
                    self.colmap[si, gene_position[int(g)]] = col
        self.colmap = torch.as_tensor(self.colmap, device=self.device)
        self.group_for_gene = torch.as_tensor(group_of, device=self.device)
        self.covered = group_of >= 0

    @torch.no_grad()
    def retrieve(self, H, k, tie_padding=64):
        require(k == 50, 'The reference bank uses 50 neighbors')
        require(hasattr(self, 'groups'), 'Call configure(panel) first')
        if not len(H) or not self.groups:
            return np.full((len(H), len(self.panel)), np.nan, np.float32), {
                'max_retained_threshold_ties': 0, 'coverage_groups': len(self.groups),
                'covered_genes': int(self.covered.sum())}
        Q = original_keys(H, self.device)
        triples = []
        # Documented exact fp16 cosine, no PCA256/ANN candidate omissions.
        for si, E in enumerate(self.keys):
            vs, ids = [], []
            for begin in range(0, len(H), 256):
                score = (Q[begin:begin+256] @ E.T).float()
                v, i = score.topk(min(k, len(E)), dim=1)
                vs.append(v)
                ids.append(i + self.offsets[si])
            triples.append((torch.cat(vs), torch.cat(ids)))
        neighbours, weights, counts = [], [], []
        width = k + tie_padding
        for hits, _ in self.groups:
            scores = torch.cat([triples[si][0] for si in hits], dim=1)
            ids = torch.cat([triples[si][1] for si in hits], dim=1)
            effective_k = min(k, scores.shape[1])
            selected, _ = scores.topk(effective_k, dim=1)
            theta = selected[:, -1:]
            mx = selected[:, :1]
            n_gt = (scores > theta).sum(1, keepdim=True)
            n_eq = (scores == theta).sum(1, keepdim=True)
            fractional = (effective_k - n_gt) / n_eq
            denominator = torch.exp((selected - mx) / TAU).sum(1, keepdim=True)
            active = (scores >= theta).sum(1)
            counts.append(active.max())
            take = min(width, scores.shape[1])
            v, order = scores.topk(take, dim=1)
            cells = torch.gather(ids, 1, order)
            mass = torch.where(v > theta, 1., torch.where(v == theta, fractional, 0.))
            w = torch.exp((v - mx) / TAU) * mass / denominator
            if take < width:
                cells = F.pad(cells, (0, width-take), value=int(self.offsets[hits[0]]))
                w = F.pad(w, (0, width-take))
            neighbours.append(cells)
            weights.append(w)
        required = int(torch.stack(counts).max().item())
        if required > width:
            # Retry to include all ties within the documented retained union.
            return self.retrieve(H, k, tie_padding=required-k)
        cells = torch.stack(neighbours, dim=1)[:, :, :required]
        w = torch.stack(weights, dim=1)[:, :, :required]
        if not torch.allclose(w.sum(2), torch.ones_like(w.sum(2)), atol=2e-5, rtol=2e-5):
            raise RuntimeError('Fractional tie weights do not sum to one')
        output = torch.full((len(H), len(self.panel)), float('nan'), device=self.device, dtype=torch.float32)
        # Gene chunks can skip entirely unsupported genes; no zero fill.
        for begin in range(0, len(self.panel), 128):
            positions = torch.arange(begin, min(begin+128, len(self.panel)), device=self.device)
            positions = positions[self.group_for_gene[positions] >= 0]
            if not len(positions):
                continue
            groups = self.group_for_gene[positions]
            # Keep each transient gather below roughly 64 MiB of float32 values.
            qstep = max(1, min(256, 16_000_000 // max(1, len(positions) * required)))
            for q0 in range(0, len(H), qstep):
                q1 = min(q0 + qstep, len(H))
                chosen = cells[q0:q1, groups, :]
                source = self.source[chosen].long()
                col = self.colmap[source, positions[None, :, None]].long()
                addresses = self.row_start[chosen] + col
                expression = self.pool[addresses]
                output[q0:q1, positions] = (w[q0:q1, groups, :] * expression).sum(2)
        return output.cpu().numpy(), {'max_retained_threshold_ties': required,
                                      'coverage_groups': len(self.groups),
                                      'covered_genes': int(self.covered.sum())}
