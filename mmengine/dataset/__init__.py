import bisect
import collections
import copy
import os
import os.path as osp
import pickle
from typing import (Any, Callable, Dict, Iterable, List, Optional,
                    Sequence, Tuple, Union)

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset as _TorchConcatDataset


class Compose:
    """Compose multiple transforms sequentially."""

    def __init__(self, transforms: Sequence):
        self.transforms = []
        for t in transforms:
            if callable(t):
                self.transforms.append(t)
            elif isinstance(t, dict):
                from mmengine.registry import TRANSFORMS
                obj = TRANSFORMS.build(t)
                self.transforms.append(obj)
            else:
                raise TypeError(f'transform must be callable or dict, got {type(t)}')

    def __call__(self, data: dict) -> Optional[dict]:
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'transforms={self.transforms})')


def force_full_init(fn):
    """Decorator to force full initialization of a dataset before calling fn."""
    def wrapper(self, *args, **kwargs):
        if not self._fully_initialized:
            self.full_init()
        return fn(self, *args, **kwargs)
    wrapper.__wrapped__ = fn
    return wrapper


class BaseDataset(Dataset):
    """Base class for datasets, mimicking mmengine.dataset.BaseDataset."""

    METAINFO: dict = {}

    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 data_prefix: dict = dict(),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        self.ann_file = ann_file
        self.data_root = data_root
        self.data_prefix = copy.deepcopy(data_prefix)
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self._fully_initialized = False

        # Merge metainfo
        self._metainfo: dict = copy.deepcopy(self.__class__.METAINFO)
        if metainfo is not None:
            self._metainfo.update(metainfo)

        self.pipeline = Compose(pipeline)

        if not lazy_init:
            self.full_init()

    def full_init(self):
        if self._fully_initialized:
            return
        # Load data list
        self.data_list = self.load_data_list()
        # Filter
        self.data_list = self.filter_data()
        # Apply indices
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)
        # Serialize if needed
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()
        self._fully_initialized = True

    def _serialize_data(self):
        data_bytes = []
        data_address = []
        offset = 0
        for item in self.data_list:
            b = pickle.dumps(item)
            data_bytes.append(b)
            data_address.append((offset, len(b)))
            offset += len(b)
        return b''.join(data_bytes), data_address

    def _get_unserialized_subset(self, indices):
        if isinstance(indices, int):
            indices = list(range(indices))
        return [self.data_list[i] for i in indices]

    def load_data_list(self) -> List[dict]:
        raise NotImplementedError

    def filter_data(self) -> List[dict]:
        return self.data_list

    @force_full_init
    def __len__(self) -> int:
        return len(self.data_list)

    @force_full_init
    def __getitem__(self, idx: int) -> dict:
        if not self._fully_initialized:
            raise RuntimeError('Dataset not fully initialized')
        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            if data is None:
                idx = (idx + 1) % len(self)
                continue
            return data
        raise Exception(f'Failed to get valid data after {self.max_refetch} retries')

    def prepare_data(self, idx: int) -> Optional[dict]:
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)

    def get_data_info(self, idx: int) -> dict:
        if self.serialize_data:
            start, length = self.data_address[idx]
            return copy.deepcopy(pickle.loads(
                self.data_bytes[start:start + length]))
        return copy.deepcopy(self.data_list[idx])

    @property
    def metainfo(self) -> dict:
        return copy.deepcopy(self._metainfo)

    def get_subset(self, indices) -> 'BaseDataset':
        sub = copy.copy(self)
        sub.data_list = [self.data_list[i] for i in indices]
        if self.serialize_data:
            sub.data_bytes, sub.data_address = sub._serialize_data()
        return sub


class ConcatDataset(_TorchConcatDataset):
    """Concatenate multiple datasets with shared metainfo."""

    def __init__(self, datasets, verify_meta: bool = True):
        super().__init__(datasets)
        self._metainfo = {}
        if datasets:
            self._metainfo = datasets[0].metainfo

    @property
    def metainfo(self) -> dict:
        return copy.deepcopy(self._metainfo)


# ---- Samplers ----

class DefaultSampler(torch.utils.data.Sampler):
    """Default sampler supporting shuffling and distributed training."""

    def __init__(self, dataset, shuffle: bool = True, seed: int = 0,
                 round_up: bool = True):
        rank, world_size = 0, 1
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
        except Exception:
            pass

        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        self.num_samples = len(dataset)
        per_rank = (self.num_samples + world_size - 1) // world_size
        self.total_size = per_rank * world_size if round_up else self.num_samples
        self.per_rank = per_rank

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        # Pad to total_size
        indices += indices[:(self.total_size - len(indices))]
        # Slice for this rank
        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices[:self.per_rank])

    def __len__(self) -> int:
        return self.per_rank

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class InfiniteSampler(torch.utils.data.Sampler):
    """Infinite sampler for iteration-based training."""

    def __init__(self, dataset, shuffle: bool = True, seed: int = 0):
        rank, world_size = 0, 1
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
        except Exception:
            pass

        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        epoch = 0
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + epoch)
            if self.shuffle:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))
            for idx in indices[self.rank::self.world_size]:
                yield idx
            epoch += 1

    def __len__(self) -> int:
        return len(self.dataset) // self.world_size


# ---- sampler sub-module alias ----
class _SamplerModule:
    DefaultSampler = DefaultSampler
    InfiniteSampler = InfiniteSampler


sampler = _SamplerModule()


__all__ = [
    'Compose', 'force_full_init', 'BaseDataset', 'ConcatDataset',
    'DefaultSampler', 'InfiniteSampler',
]
