"""Image and annotation loading transforms (PIL/numpy-based, no cv2 required)."""

import os
import os.path as osp
from typing import Dict, Optional, Union

import numpy as np
from PIL import Image

from .base import BaseTransform


def to_tensor(data):
    """Convert numpy/PIL data to torch tensor."""
    import torch
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(data))
    if isinstance(data, (int, float)):
        return torch.tensor(data)
    raise TypeError(f'Cannot convert {type(data)} to tensor')


class LoadImageFromFile(BaseTransform):
    """Load an image from file using PIL (no cv2 required)."""

    def __init__(self, to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'pillow',
                 backend_args: Optional[dict] = None,
                 ignore_empty: bool = False):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.backend_args = backend_args
        self.ignore_empty = ignore_empty

    def transform(self, results: Dict) -> Optional[Dict]:
        filename = results.get('img_path', results.get('filename', ''))
        try:
            img = Image.open(filename)
            if self.color_type == 'color':
                img = img.convert('RGB')
                img_array = np.array(img)  # H, W, C (RGB)
                # Convert to BGR for compatibility with mmcv convention
                img_array = img_array[:, :, ::-1].copy()
            elif self.color_type == 'grayscale':
                img = img.convert('L')
                img_array = np.array(img)
            else:
                img_array = np.array(img)
        except Exception as e:
            if self.ignore_empty:
                return None
            raise IOError(f'Failed to load image {filename}: {e}')

        if self.to_float32:
            img_array = img_array.astype(np.float32)

        results['img'] = img_array
        results['img_shape'] = img_array.shape[:2]
        results['ori_shape'] = img_array.shape[:2]
        return results


class LoadAnnotations(BaseTransform):
    """Load semantic segmentation annotations."""

    def __init__(self, with_bbox: bool = False, with_label: bool = False,
                 with_seg: bool = True, with_keypoints: bool = False,
                 imdecode_backend: str = 'pillow',
                 backend_args: Optional[dict] = None, **kwargs):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_seg = with_seg
        self.with_keypoints = with_keypoints
        self.imdecode_backend = imdecode_backend
        self.backend_args = backend_args

    def _load_seg_map(self, results: Dict) -> Dict:
        seg_map_path = results.get('seg_map_path', '')
        try:
            gt_img = Image.open(seg_map_path)
            gt_array = np.array(gt_img)
        except Exception as e:
            raise IOError(f'Failed to load seg map {seg_map_path}: {e}')
        results['gt_seg_map'] = gt_array
        results['seg_fields'] = results.get('seg_fields', [])
        results['seg_fields'].append('gt_seg_map')
        return results

    def transform(self, results: Dict) -> Optional[Dict]:
        if self.with_seg:
            results = self._load_seg_map(results)
        return results


__all__ = ['LoadImageFromFile', 'LoadAnnotations', 'to_tensor']
