"""Retain Stage-1 tail batches, masking DDP sampler padding from supervision."""
from itertools import islice
from torch.utils.data import Dataset, DistributedSampler


class TailSampler(DistributedSampler):
    def __iter__(self):
        for slot,index in enumerate(super().__iter__()):
            yield index,slot*self.num_replicas+self.rank<len(self.dataset)


class ResumeTailSampler(TailSampler):
    """Skip sampler slots before DataLoader can decode the corresponding images."""
    start_slot = 0

    def set_start(self, start_slot):
        if not 0 <= start_slot <= self.num_samples:
            raise ValueError('Resume sampler cursor outside this rank')
        self.start_slot = int(start_slot)

    def __len__(self):
        return self.num_samples-self.start_slot

    def __iter__(self):
        # DistributedSampler creates the identical seed+epoch permutation. islice
        # skips its list iterator in C, without calling Dataset.__getitem__.
        indices = DistributedSampler.__iter__(self)
        for slot, index in enumerate(islice(indices, self.start_slot, None), self.start_slot):
            yield index, slot*self.num_replicas+self.rank < len(self.dataset)


class TailDataset(Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,index):
        if isinstance(index,tuple):index,valid=index
        else:valid=True
        return (*self.base[index],valid)
