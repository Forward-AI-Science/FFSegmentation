import os
import os.path as osp
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


class BaseVisBackend:
    """Base class for visualization backends."""

    def __init__(self, save_dir: str = '', **kwargs):
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def add_image(self, name: str, image: np.ndarray, step: int = 0, **kwargs):
        pass

    def add_scalar(self, name: str, value, step: int = 0, **kwargs):
        pass

    def add_scalars(self, scalar_dict: dict, step: int = 0, **kwargs):
        pass


class LocalVisBackend(BaseVisBackend):
    """Saves visualizations to local disk."""

    def add_image(self, name: str, image: np.ndarray, step: int = 0, **kwargs):
        if self.save_dir:
            try:
                from PIL import Image
                filepath = osp.join(self.save_dir, f'{name}_{step}.png')
                img = Image.fromarray(image.astype(np.uint8))
                img.save(filepath)
            except Exception as e:
                warnings.warn(f'Failed to save image: {e}')


class Visualizer:
    """Global visualizer with multiple backends."""

    _instance: Optional['Visualizer'] = None

    def __init__(self, name: str = 'visualizer',
                 vis_backends: Optional[List[dict]] = None,
                 save_dir: str = '', **kwargs):
        self.name = name
        self.save_dir = save_dir
        self._vis_backends: Dict[str, BaseVisBackend] = {}
        self._image: Optional[np.ndarray] = None

        if vis_backends:
            for backend_cfg in vis_backends:
                cfg = backend_cfg.copy()
                btype = cfg.pop('type', 'LocalVisBackend')
                if btype == 'LocalVisBackend':
                    be = LocalVisBackend(save_dir=save_dir, **cfg)
                else:
                    be = BaseVisBackend(save_dir=save_dir, **cfg)
                self._vis_backends[btype] = be

        Visualizer._instance = self

    @classmethod
    def get_current_instance(cls) -> 'Visualizer':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get_instance(cls, name: str, **kwargs) -> 'Visualizer':
        if cls._instance is None:
            cls._instance = cls(name=name, **kwargs)
        return cls._instance

    def set_image(self, image: np.ndarray):
        self._image = image.copy()

    def get_image(self) -> Optional[np.ndarray]:
        return self._image

    def draw_sem_seg(self, image: np.ndarray, sem_seg, classes=None,
                     palette=None, **kwargs) -> 'Visualizer':
        self._image = image.copy()
        return self

    def add_image(self, name: str, image: np.ndarray, step: int = 0, **kwargs):
        for backend in self._vis_backends.values():
            backend.add_image(name, image, step, **kwargs)

    def add_scalar(self, name: str, value, step: int = 0, **kwargs):
        for backend in self._vis_backends.values():
            backend.add_scalar(name, value, step)

    def add_scalars(self, scalar_dict: dict, step: int = 0, **kwargs):
        for backend in self._vis_backends.values():
            backend.add_scalars(scalar_dict, step)

    def show(self, win_name: str = '', wait_time: float = 0.):
        pass


class SegLocalVisualizer(Visualizer):
    """Segmentation-specific visualizer."""

    def __init__(self, name: str = 'visualizer',
                 vis_backends: Optional[List[dict]] = None,
                 save_dir: str = '', alpha: float = 0.8, **kwargs):
        super().__init__(name=name, vis_backends=vis_backends,
                         save_dir=save_dir, **kwargs)
        self.alpha = alpha


__all__ = ['Visualizer', 'LocalVisBackend', 'BaseVisBackend', 'SegLocalVisualizer']
