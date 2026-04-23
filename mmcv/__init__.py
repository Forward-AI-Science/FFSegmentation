"""Pure-PyTorch replacement for mmcv.
Compatible with PyTorch 2.11+ and Python 3.12+.
"""

__version__ = '2.1.0'
version_info = (2, 1, 0)

from .image import imread, imwrite, imshow, bgr2rgb, rgb2bgr

from .cnn import (ConvModule, DepthwiseSeparableConvModule, ContextBlock,
                  NonLocal2d, build_conv_layer, build_norm_layer,
                  build_activation_layer, build_upsample_layer,
                  build_plugin_layer, DropPath, Conv2dAdaptivePadding,
                  Linear, Scale, Conv2d)
from .transforms import (BaseTransform, Compose, to_tensor,
                         LoadImageFromFile, LoadAnnotations,
                         Resize, RandomResize, RandomFlip)
from .ops import (DeformConv2d, ModulatedDeformConv2d,
                  CrissCrossAttention, PSAMask,
                  point_sample, sigmoid_focal_loss)

__all__ = [
    '__version__', 'version_info',
    'imread', 'imwrite', 'imshow', 'bgr2rgb', 'rgb2bgr',
    'ConvModule', 'DepthwiseSeparableConvModule', 'ContextBlock', 'NonLocal2d',
    'build_conv_layer', 'build_norm_layer', 'build_activation_layer',
    'build_upsample_layer', 'build_plugin_layer', 'DropPath',
    'Conv2dAdaptivePadding', 'Linear', 'Scale', 'Conv2d',
    'BaseTransform', 'Compose', 'to_tensor',
    'LoadImageFromFile', 'LoadAnnotations',
    'Resize', 'RandomResize', 'RandomFlip',
    'DeformConv2d', 'ModulatedDeformConv2d',
    'CrissCrossAttention', 'PSAMask', 'point_sample', 'sigmoid_focal_loss',
]
