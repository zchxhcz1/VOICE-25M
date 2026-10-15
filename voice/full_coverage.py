"""Spatial ownership with shuffled, epoch-specific partitions and bounded halos."""
import os
import zlib
import numpy as np
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
import torch
from voice.training_cache import cached_arrays, array_stamp


REVISION='full_coverage_halo_v1'


def partition_cells(pos, target_cells, seed, epoch):
    if target_cells<1 or len(pos)<1 or not np.isfinite(pos).all():
        raise ValueError('Positive target budget and finite nonempty positions required')
    rng=np.random.RandomState((seed+epoch*1000003)%(2**31-1))
    angle=rng.uniform(-np.pi/4,np.pi/4)
    rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    p=np.asarray(pos,np.float64)@rot
    span=np.maximum(np.ptp(p,axis=0),1.)
    side=max(float(np.sqrt(np.prod(span)*target_cells*2/len(p))),1.)
    origin=p.min(0)-rng.uniform(0,side,2)
    grid=np.floor((p-origin)/side).astype(np.int64)
    order=np.lexsort((np.arange(len(p)),grid[:,1],grid[:,0]))
    groups=grid[order];cuts=np.r_[0,np.flatnonzero(np.any(groups[1:]!=groups[:-1],axis=1))+1,len(p)]
    blocks=[]
    def split(ids):
        if len(ids)<=target_cells:
            blocks.append(ids);return
        axis=int(np.argmax(np.ptp(p[ids],axis=0)))
        ids=ids[np.lexsort((ids,p[ids,axis]))]
        mid=len(ids)//2
        split(ids[:mid]);split(ids[mid:])
    for lo,hi in zip(cuts[:-1],cuts[1:]):split(order[lo:hi])
    flat=np.concatenate(blocks).astype(np.int64)
    ptr=np.r_[0,np.cumsum([len(b) for b in blocks])].astype(np.int64)
    counts=np.bincount(flat,minlength=len(p))
    if not np.all(counts==1):raise ValueError('Spatial partition must own every target exactly once')
    return flat,ptr


def halo_cells(pos, targets, tree, context_cells, radius):
    if context_cells<1 or len(targets)==len(pos):return np.empty(0,np.int64)
    dist,ids=tree.query(pos[targets],k=min(16,len(pos)))
    if ids.ndim==1:ids=ids[:,None];dist=dist[:,None]
    # First outside neighbour of each target precedes more distant neighbours.
    valid=(dist<=radius)&~np.isin(ids,targets)
    candidates=ids.T[valid.T]
    if not len(candidates):return np.empty(0,np.int64)
    _,first=np.unique(candidates,return_index=True)
    return candidates[np.sort(first)][:context_cells].astype(np.int64)


class CoveragePlan:
    def __init__(self,stores,epochs,target_cells,seed,world,cache=None):
        if epochs<1 or world<1:raise ValueError('Positive epochs/world required')
        self.stores=stores;self.world=world;self.epochs=[];self.step_ends=[]
        self.cells=sum(s.n for s in stores)
        total=0
        for epoch in range(epochs):
            plans=[]
            for s in stores:
                object_seed=(seed+zlib.crc32(s.obj.encode()))%(2**31-1)
                signature={'revision':REVISION,'pos':array_stamp(s.pos),'seed':object_seed,
                           'epoch':epoch,'target_cells':target_cells}
                def build(root):
                    order,ptr=partition_cells(s.pos,target_cells,object_seed,epoch)
                    np.save(root/'order.tmp.npy',order);np.save(root/'ptr.tmp.npy',ptr)
                    return ['order','ptr']
                if cache:
                    arrays=cached_arrays(f'{cache}/coverage/{s.obj}/epoch{epoch}',signature,build)
                else:
                    order,ptr=partition_cells(s.pos,target_cells,object_seed,epoch)
                    arrays={'order':order,'ptr':ptr}
                plans.append(arrays)
            ends=np.cumsum([len(p['ptr'])-1 for p in plans])
            n=int(ends[-1]);per_rank=(n+world-1)//world
            shuffle=np.random.RandomState(seed+epoch*7919).permutation(n)
            sizes=np.concatenate([np.diff(p['ptr']) for p in plans])
            self.epochs.append({'plans':plans,'ends':ends,'shuffle':shuffle,'n':n,'steps':per_rank,'sizes':sizes})
            total+=per_rank;self.step_ends.append(total)
        self.total_steps=total
        self.blocks=sum(e['n'] for e in self.epochs)

    def locate(self,step,rank):
        if not 0<=step<self.total_steps or not 0<=rank<self.world:raise IndexError((step,rank))
        epoch=int(np.searchsorted(self.step_ends,step,side='right'))
        e=self.epochs[epoch];local=step-(self.step_ends[epoch-1] if epoch else 0)
        slot=local*self.world+rank;valid=slot<e['n']
        block=int(e['shuffle'][slot if valid else 0])
        obj=int(np.searchsorted(e['ends'],block,side='right'))
        local_block=block-(int(e['ends'][obj-1]) if obj else 0)
        p=e['plans'][obj];targets=p['order'][p['ptr'][local_block]:p['ptr'][local_block+1]]
        return epoch,obj,targets,valid

    def supervised_before(self,step,rank):
        if not 0<=step<=self.total_steps or not 0<=rank<self.world:raise IndexError((step,rank))
        count=0;start=0
        for e in self.epochs:
            used=min(max(step-start,0),e['steps'])
            slots=np.arange(rank,min(e['n'],used*self.world),self.world)
            count+=int(e['sizes'][e['shuffle'][slots]].sum())
            start+=e['steps']
        return count


class FullCoverageDataset(Dataset):
    def __init__(self,plan,rank,context_cells,radius,seed,d4,start=0):
        self.plan=plan;self.rank=rank;self.context_cells=context_cells;self.radius=radius
        self.seed=seed;self.d4=d4;self.start=start;self.trees={}
    def __len__(self):return self.plan.total_steps-self.start
    def __getitem__(self,i):
        step=i+self.start
        epoch,obj,targets,valid=self.plan.locate(step,self.rank)
        s=self.plan.stores[obj]
        if obj not in self.trees:self.trees[obj]=cKDTree(s.pos)
        context=halo_cells(s.pos,targets,self.trees[obj],self.context_cells,self.radius)
        idx=np.r_[targets,context]
        mask=np.arange(len(idx))<len(targets)
        if not valid:mask[:]=False
        rng=np.random.RandomState((self.seed+step*1000003+self.rank*10007)%(2**31-1))
        rot,flip=(rng.randint(4),bool(rng.randint(2))) if self.d4 else (0,False)
        img,W=s.load(idx,rot,flip)
        return (torch.from_numpy(img),torch.from_numpy(W),torch.from_numpy(s.y[idx].astype(np.float32)),
                torch.from_numpy(s.area_abs[idx]),torch.from_numpy(s.area_rel[idx]),torch.from_numpy(s.pos[idx]),
                int(s.plat),torch.from_numpy(s.head_idx),torch.from_numpy(s.is_aux.astype(np.float32)),
                torch.from_numpy(mask))


def supervised_sums(y,mu,log_theta,wg,mask,nb_nll):
    """Context labels do not enter either loss; return sums over owned target cells."""
    n=int(mask.sum().item())
    if n==0:
        zero=mu.sum()*0+log_theta.sum()*0
        return zero,zero,0
    y=y[mask];mu=mu[mask]
    nll=(nb_nll(y,mu,log_theta)*wg).sum()/wg.sum()
    mse=((torch.log1p(mu)-torch.log1p(y))**2*wg).sum()/wg.sum()
    return nll,mse,n
