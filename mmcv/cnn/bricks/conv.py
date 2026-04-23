import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_conv_layer(cfg: Optional[dict], *args, **kwargs) -> nn.Module:
    """Build conv layer from config dict."""
    if cfg is None:
        return nn.Conv2d(*args, **kwargs)
    cfg = cfg.copy()
    conv_type = cfg.pop('type', 'Conv2d')
    conv_map = {
        'Conv2d': nn.Conv2d,
        'Conv': nn.Conv2d,
        'Conv1d': nn.Conv1d,
        'Conv3d': nn.Conv3d,
        'ConvTranspose2d': nn.ConvTranspose2d,
        'AdaptivePad': Conv2dAdaptivePadding,
    }
    cls = conv_map.get(conv_type)
    if cls is None:
        # Try as plain nn attribute
        cls = getattr(nn, conv_type, None)
    if cls is None:
        raise KeyError(f'Unknown conv type: {conv_type}')
    return cls(*args, **cfg, **kwargs)


class Conv2dAdaptivePadding(nn.Conv2d):
    """Conv2d with adaptive (same) padding."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.shape[-2:]
        kh, kw = self.weight.shape[-2:]
        sh, sw = self.stride
        oh = math.ceil(ih / sh)
        ow = math.ceil(iw / sw)
        pad_h = max((oh - 1) * sh + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                          pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, 0,
                        self.dilation, self.groups)


__all__ = ['build_conv_layer', 'Conv2dAdaptivePadding']
