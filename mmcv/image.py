"""PIL-based image I/O replacing mmcv.image (no cv2 required)."""

import os
from typing import Optional, Union

import numpy as np
from PIL import Image


def imread(img: Union[str, np.ndarray],
           flag: str = 'color',
           channel_order: str = 'bgr',
           backend: Optional[str] = None) -> np.ndarray:
    """Read an image file and return as numpy array.

    Args:
        img: File path or numpy array (returned as-is).
        flag: 'color', 'grayscale', or 'unchanged'.
        channel_order: 'bgr' or 'rgb'. Only used when flag='color'.
        backend: Ignored (always uses PIL).

    Returns:
        np.ndarray: Image array.
    """
    if isinstance(img, np.ndarray):
        return img

    pil_img = Image.open(str(img))

    if flag == 'grayscale':
        pil_img = pil_img.convert('L')
        return np.array(pil_img)
    elif flag == 'unchanged':
        return np.array(pil_img)
    else:  # 'color'
        pil_img = pil_img.convert('RGB')
        arr = np.array(pil_img)
        if channel_order == 'bgr':
            arr = arr[:, :, ::-1].copy()
        return arr


def imwrite(img: np.ndarray, file_path: str,
            params=None, auto_mkdir: bool = True) -> bool:
    """Write image to file using PIL.

    Args:
        img: Image as numpy array (BGR or RGB).
        file_path: Output file path.
        params: Ignored.
        auto_mkdir: Create parent directories if missing.

    Returns:
        bool: True on success.
    """
    if auto_mkdir:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    # Assume BGR if 3-channel, convert to RGB for PIL
    if img.ndim == 3 and img.shape[2] == 3:
        img = img[:, :, ::-1]  # BGR → RGB
    pil_img = Image.fromarray(img.astype(np.uint8))
    pil_img.save(file_path)
    return True


def imshow(img: np.ndarray, win_name: str = '', wait_time: int = 0):
    """Display image (no-op in headless environments)."""
    pass


def bgr2rgb(img: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB."""
    return img[:, :, ::-1].copy()


def rgb2bgr(img: np.ndarray) -> np.ndarray:
    """Convert an RGB image to BGR."""
    return img[:, :, ::-1].copy()


__all__ = ['imread', 'imwrite', 'imshow', 'bgr2rgb', 'rgb2bgr']
