import copy
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import torch


class BaseDataElement:
    """A base data structure interface mimicking mmengine.structures.BaseDataElement.

    Stores two kinds of data:
      - metainfo: auxiliary meta information (stored in _metainfo_fields)
      - data fields: tensors / arrays (stored as regular attributes, tracked in _data_fields)
    """

    def __init__(self, *, metainfo: Optional[Dict] = None, **kwargs):
        self._metainfo_fields: Set[str] = set()
        self._data_fields: Set[str] = set()
        if metainfo is not None:
            self.set_metainfo(metainfo)
        if kwargs:
            self.set_data(kwargs)

    # ---- metainfo ----
    def set_metainfo(self, metainfo: Dict):
        for k, v in metainfo.items():
            object.__setattr__(self, k, v)
            self._metainfo_fields.add(k)
            self._data_fields.discard(k)

    def metainfo_keys(self) -> List[str]:
        return list(self._metainfo_fields)

    def metainfo_values(self) -> List:
        return [getattr(self, k) for k in self._metainfo_fields]

    def metainfo_items(self) -> Iterator[Tuple[str, Any]]:
        for k in self._metainfo_fields:
            yield k, getattr(self, k)

    @property
    def metainfo(self) -> Dict:
        return {k: getattr(self, k) for k in self._metainfo_fields}

    # ---- data ----
    def set_data(self, data: Dict):
        for k, v in data.items():
            object.__setattr__(self, k, v)
            self._data_fields.add(k)
            self._metainfo_fields.discard(k)

    def set_field(self, value, name: str, dtype=None):
        if dtype is not None and not isinstance(value, dtype):
            raise TypeError(
                f'Field {name} must be of type {dtype}, got {type(value)}')
        object.__setattr__(self, name, value)
        self._data_fields.add(name)
        self._metainfo_fields.discard(name)

    def keys(self) -> List[str]:
        return list(self._data_fields)

    def values(self) -> List:
        return [getattr(self, k) for k in self._data_fields]

    def items(self) -> Iterator[Tuple[str, Any]]:
        for k in self._data_fields:
            yield k, getattr(self, k)

    def __contains__(self, item: str) -> bool:
        return item in self._data_fields or item in self._metainfo_fields

    def __getitem__(self, key: str):
        if key in self._data_fields or key in self._metainfo_fields:
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any):
        if key in self._metainfo_fields:
            self.set_metainfo({key: value})
        else:
            self.set_data({key: value})

    def __setattr__(self, name: str, value: Any):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)
            # Auto-track in data fields unless it's metainfo
            if name not in self._metainfo_fields:
                self._data_fields.add(name)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        meta = {k: getattr(self, k) for k in self._metainfo_fields}
        data = {k: getattr(self, k) for k in self._data_fields}
        return f'<{cls}(metainfo={meta}, data_fields={list(data.keys())})>'

    # ---- device / tensor utilities ----
    def to(self, *args, **kwargs) -> 'BaseDataElement':
        new = copy.copy(self)
        for k in self._data_fields:
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                object.__setattr__(new, k, v.to(*args, **kwargs))
            elif isinstance(v, BaseDataElement):
                object.__setattr__(new, k, v.to(*args, **kwargs))
        return new

    def cpu(self) -> 'BaseDataElement':
        return self.to('cpu')

    def cuda(self, device=None) -> 'BaseDataElement':
        return self.to('cuda' if device is None else f'cuda:{device}')

    def numpy(self) -> 'BaseDataElement':
        new = copy.copy(self)
        for k in self._data_fields:
            v = getattr(self, k)
            if isinstance(v, torch.Tensor):
                object.__setattr__(new, k, v.detach().cpu().numpy())
        return new

    def clone(self) -> 'BaseDataElement':
        new = copy.deepcopy(self)
        return new


class PixelData(BaseDataElement):
    """Pixel-level data container.

    Can hold a primary tensor under the key 'data' plus metainfo.
    Supports shape property.
    """

    def __init__(self, *, metainfo: Optional[Dict] = None, **kwargs):
        super().__init__(metainfo=metainfo)
        if kwargs:
            self.set_data(kwargs)

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        if 'data' in self._data_fields:
            t = getattr(self, 'data')
            if isinstance(t, torch.Tensor):
                return tuple(t.shape[-2:])  # H, W
        return None

    def __setattr__(self, name: str, value: Any):
        super().__setattr__(name, value)


class InstanceData(BaseDataElement):
    """Instance-level data container (e.g. bounding boxes, masks)."""

    def __len__(self) -> int:
        for k in self._data_fields:
            v = getattr(self, k)
            if hasattr(v, '__len__'):
                return len(v)
        return 0


__all__ = ['BaseDataElement', 'PixelData', 'InstanceData']
