"""Geometric and photometric transforms (PIL/numpy-based)."""

import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from .base import BaseTransform
from .utils import cache_randomness


class Resize(BaseTransform):
    """Resize image and annotations."""

    def __init__(self, scale: Union[Tuple[int, int], int] = None,
                 scale_factor: Optional[float] = None,
                 keep_ratio: bool = False,
                 clip_object_border: bool = True,
                 backend: str = 'pillow',
                 interpolation: str = 'bilinear', **kwargs):
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        self.backend = backend
        self.interpolation = interpolation

        if scale is not None:
            if isinstance(scale, int):
                self.scale = (scale, scale)
            else:
                self.scale = scale  # (h, w) or (w, h)? mmcv uses (w, h)
        else:
            self.scale = None
        self.scale_factor = scale_factor

    def _resize_img(self, img: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
        h, w = new_size  # (H, W)
        pil_img = Image.fromarray(img.astype(np.uint8) if img.dtype == np.uint8 else img)
        pil_img = pil_img.resize((w, h), Image.BILINEAR)
        return np.array(pil_img)

    def _resize_seg(self, seg: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
        h, w = new_size
        pil_seg = Image.fromarray(seg.astype(np.uint8))
        pil_seg = pil_seg.resize((w, h), Image.NEAREST)
        return np.array(pil_seg)

    def transform(self, results: Dict) -> Optional[Dict]:
        img = results.get('img')
        if img is None:
            return results

        h, w = img.shape[:2]

        if self.scale is not None:
            new_w, new_h = self.scale  # mmcv convention: (w, h)
            if self.keep_ratio:
                scale_factor = min(new_h / h, new_w / w)
                new_h = int(h * scale_factor + 0.5)
                new_w = int(w * scale_factor + 0.5)
        elif self.scale_factor is not None:
            new_h = int(h * self.scale_factor + 0.5)
            new_w = int(w * self.scale_factor + 0.5)
        else:
            return results

        results['img'] = self._resize_img(img, (new_h, new_w))
        results['img_shape'] = (new_h, new_w)
        results['scale_factor'] = (new_w / w, new_h / h)

        for key in results.get('seg_fields', []):
            results[key] = self._resize_seg(results[key], (new_h, new_w))

        return results


class RandomResize(BaseTransform):
    """Random resize with scale jitter."""

    def __init__(self, scale: Union[Tuple, List], ratio_range: Tuple[float, float] = None,
                 keep_ratio: bool = True, resize_type: str = 'Resize',
                 **resize_kwargs):
        self.scale = scale
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self._resize_cfg = dict(type=resize_type, keep_ratio=keep_ratio,
                                **resize_kwargs)

    @cache_randomness
    def _random_scale(self, img_shape: Tuple[int, int]) -> Tuple[int, int]:
        if self.ratio_range is not None:
            min_ratio, max_ratio = self.ratio_range
            ratio = random.uniform(min_ratio, max_ratio)
            if isinstance(self.scale, (list, tuple)) and len(self.scale) == 2:
                if isinstance(self.scale[0], int):
                    scale = self.scale
                else:
                    scale = random.choice(self.scale)
            else:
                scale = self.scale
            if isinstance(scale, int):
                scale = (scale, scale)
            return (int(scale[0] * ratio), int(scale[1] * ratio))
        if isinstance(self.scale, (list, tuple)) and len(self.scale) == 2:
            if isinstance(self.scale[0], (list, tuple)):
                return random.choice(self.scale)
        return self.scale

    def transform(self, results: Dict) -> Optional[Dict]:
        img = results.get('img')
        if img is None:
            return results
        scale = self._random_scale(img.shape[:2])
        resize = Resize(scale=scale, keep_ratio=self.keep_ratio)
        return resize(results)


class RandomFlip(BaseTransform):
    """Random flip transform."""

    def __init__(self, prob: float = 0.5,
                 direction: Union[str, List[str]] = 'horizontal'):
        self.prob = prob
        if isinstance(direction, str):
            direction = [direction]
        self.direction = direction

    @cache_randomness
    def _random_direction(self) -> Optional[str]:
        if random.random() < self.prob:
            return random.choice(self.direction)
        return None

    def transform(self, results: Dict) -> Optional[Dict]:
        cur_dir = self._random_direction()
        if cur_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
            return results

        results['flip'] = True
        results['flip_direction'] = cur_dir

        img = results.get('img')
        if img is not None:
            if cur_dir == 'horizontal':
                results['img'] = img[:, ::-1, ...].copy() if img.ndim > 2 else img[:, ::-1].copy()
            elif cur_dir == 'vertical':
                results['img'] = img[::-1, ...].copy()
            elif cur_dir == 'diagonal':
                results['img'] = img[::-1, ::-1, ...].copy() if img.ndim > 2 else img[::-1, ::-1].copy()

        for key in results.get('seg_fields', []):
            seg = results[key]
            if cur_dir == 'horizontal':
                results[key] = seg[:, ::-1].copy()
            elif cur_dir == 'vertical':
                results[key] = seg[::-1, :].copy()
            elif cur_dir == 'diagonal':
                results[key] = seg[::-1, ::-1].copy()

        return results


__all__ = ['Resize', 'RandomResize', 'RandomFlip']
