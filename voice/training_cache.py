"""Shared, checked disk arrays for full-data training in interactive allocations."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def file_stamp(path):
    p=Path(path);st=p.stat()
    return [str(p.resolve()),st.st_size,st.st_mtime_ns]


def array_stamp(array):
    a=np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()


def cached_arrays(directory,signature,build):
    """One writer per sample; publish metadata last, and map completed arrays."""
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    marker=root/'metadata.json'
    with (root/'write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if marker.is_file():
            meta=json.loads(marker.read_text())
            if meta['signature']!=signature:
                raise ValueError(f'Stale training cache: {root}; use a fresh cache directory')
        else:
            arrays=build(root)
            meta={'signature':signature,'arrays':list(arrays)}
            for name in arrays:
                os.replace(root/f'{name}.tmp.npy',root/f'{name}.npy')
            tmp=root/'metadata.tmp.json';tmp.write_text(json.dumps(meta,sort_keys=True))
            os.replace(tmp,marker)
        return {name:np.load(root/f'{name}.npy',mmap_mode='c') for name in meta['arrays']}


def sparse_targets(directory,X,rows,cols,signature,chunk=2048):
    signature={**signature,'rows':array_stamp(rows),'cols':array_stamp(cols),'dtype':'float16'}
    def build(root):
        y=np.lib.format.open_memmap(root/'Y.tmp.npy',mode='w+',dtype=np.float16,shape=(len(rows),len(cols)))
        for lo in range(0,len(rows),chunk):
            hi=min(len(rows),lo+chunk)
            y[lo:hi]=X[rows[lo:hi]][:,cols].toarray().astype(np.float16)
        y.flush();del y
        return ['Y']
    return cached_arrays(directory,signature,build)['Y']


class JoinedArrays:
    """Read-only logical concatenation; slices materialize at most one statistics chunk."""
    def __init__(self,arrays):
        self.arrays=arrays;self.ends=np.cumsum([len(x) for x in arrays])
        self.shape=(int(self.ends[-1]),*arrays[0].shape[1:]);self.dtype=arrays[0].dtype
    def __len__(self):return self.shape[0]
    def __getitem__(self,index):
        if isinstance(index,slice):
            start,stop,step=index.indices(len(self))
            if step!=1:raise ValueError('Only contiguous training-statistics slices are supported')
            pieces=[];offset=0
            for a,end in zip(self.arrays,self.ends):
                lo=max(start-offset,0);hi=min(stop-offset,len(a))
                if hi>lo:pieces.append(a[lo:hi])
                offset=end
                if offset>=stop:break
            return np.concatenate(pieces) if pieces else np.empty((0,*self.shape[1:]),self.dtype)
        if index<0:index+=len(self)
        if index<0 or index>=len(self):raise IndexError(index)
        k=int(np.searchsorted(self.ends,index,side='right'))
        return self.arrays[k][index-(int(self.ends[k-1]) if k else 0)]
