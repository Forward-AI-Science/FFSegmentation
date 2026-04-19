from .base import BaseTransform
from .loading import LoadAnnotations, LoadImageFromFile, to_tensor
from .processing import RandomFlip, RandomResize, Resize
from .utils import cache_randomness

# Register base mmcv transforms with the mmengine TRANSFORMS registry
# so they can be found by dict-based pipeline configs like {'type': 'LoadImageFromFile'}
try:
    from mmengine.registry import TRANSFORMS as _TRANSFORMS
    for _cls in [LoadImageFromFile, LoadAnnotations, Resize, RandomResize, RandomFlip]:
        if _cls.__name__ not in _TRANSFORMS._module_dict:
            _TRANSFORMS.register_module(module=_cls)
except Exception:
    pass


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms):
        self.transforms = []
        for t in transforms:
            if callable(t):
                self.transforms.append(t)
            elif isinstance(t, dict):
                # Lazy build from registry
                from mmengine.registry import TRANSFORMS
                obj = TRANSFORMS.build(t)
                self.transforms.append(obj)

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(transforms={self.transforms})'


__all__ = [
    'BaseTransform', 'Compose', 'to_tensor',
    'LoadImageFromFile', 'LoadAnnotations',
    'Resize', 'RandomResize', 'RandomFlip',
    'cache_randomness',
]
