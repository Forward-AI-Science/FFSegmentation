"""mmcv.cnn: CNN building blocks in pure PyTorch."""

from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from .bricks import (build_activation_layer, build_conv_layer,
                     build_norm_layer, build_dropout, DropPath,
                     Conv2dAdaptivePadding, MultiheadAttention, FFN,
                     BaseTransformerLayer)
from .bricks.activation import build_activation_layer
from .bricks.conv import build_conv_layer, Conv2dAdaptivePadding
from .bricks.norm import build_norm_layer
from .bricks.drop import DropPath, build_dropout
from .bricks.transformer import MultiheadAttention, FFN, BaseTransformerLayer


def build_upsample_layer(cfg: Optional[dict], *args, **kwargs) -> nn.Module:
    """Build upsample layer from config."""
    if cfg is None:
        return nn.Upsample(*args, **kwargs)
    cfg = cfg.copy()
    upsample_type = cfg.pop('type', 'nearest')
    if upsample_type in ('deconv', 'ConvTranspose2d'):
        return nn.ConvTranspose2d(*args, **cfg, **kwargs)
    elif upsample_type == 'pixel_shuffle':
        upscale_factor = cfg.pop('scale_factor', 2)
        return nn.PixelShuffle(upscale_factor)
    elif upsample_type in ('bilinear', 'nearest', 'bicubic'):
        return nn.Upsample(mode=upsample_type, **cfg)
    else:
        return nn.Upsample(mode=upsample_type, **cfg)


def build_plugin_layer(cfg: dict, in_channels: int, postfix: Union[int, str] = '',
                       **kwargs) -> Tuple[str, nn.Module]:
    """Build a plugin layer."""
    cfg = cfg.copy()
    plugin_type = cfg.pop('type', '')
    # Return a passthrough if unknown
    return str(postfix), nn.Identity()


# ---- Linear alias ----
Linear = nn.Linear


# ---- Scale ----
class Scale(nn.Module):
    """Learnable scalar multiplier."""
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


# ---- Conv2d alias ----
Conv2d = nn.Conv2d


# ---- ConvModule ----
class ConvModule(nn.Module):
    """Conv + optional Norm + optional Activation, compatible with mmcv API."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: Union[bool, str] = 'auto',
                 conv_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 inplace: bool = True,
                 with_spectral_norm: bool = False,
                 padding_mode: str = 'zeros',
                 order: Tuple[str, ...] = ('conv', 'norm', 'act'),
                 init_cfg: Optional[dict] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        self.with_spectral_norm = with_spectral_norm
        self.order = order

        # Bias defaults to False if norm is used
        if bias == 'auto':
            bias = not self.with_norm

        self.conv = build_conv_layer(
            conv_cfg, in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)

        if with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # Norm layer
        if self.with_norm:
            norm_channels = out_channels if order.index('norm') > order.index('conv') else in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
        else:
            self.norm_name = None

        # Activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            if inplace and 'inplace' not in act_cfg_:
                act_cfg_['inplace'] = inplace
            self.activate = build_activation_layer(act_cfg_)
        else:
            self.activate = None

    @property
    def norm(self) -> Optional[nn.Module]:
        if self.norm_name:
            return getattr(self, self.norm_name)
        return None

    def forward(self, x: torch.Tensor,
                activate: bool = True, norm: bool = True) -> torch.Tensor:
        for layer_name in self.order:
            if layer_name == 'conv':
                x = self.conv(x)
            elif layer_name == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer_name == 'act' and activate and self.with_activation:
                x = self.activate(x)
        return x


# ---- DepthwiseSeparableConvModule ----
class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = dict(type='ReLU'),
                 dw_norm_cfg: Optional[dict] = 'default',
                 dw_act_cfg: Optional[dict] = 'default',
                 pw_norm_cfg: Optional[dict] = 'default',
                 pw_act_cfg: Optional[dict] = 'default', **kwargs):
        super().__init__()
        dw_norm_cfg = norm_cfg if dw_norm_cfg == 'default' else dw_norm_cfg
        dw_act_cfg = act_cfg if dw_act_cfg == 'default' else dw_act_cfg
        pw_norm_cfg = norm_cfg if pw_norm_cfg == 'default' else pw_norm_cfg
        pw_act_cfg = act_cfg if pw_act_cfg == 'default' else pw_act_cfg

        self.depthwise_conv = ConvModule(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, norm_cfg=dw_norm_cfg, act_cfg=dw_act_cfg)
        self.pointwise_conv = ConvModule(
            in_channels, out_channels, 1,
            norm_cfg=pw_norm_cfg, act_cfg=pw_act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


# ---- ContextBlock ----
class ContextBlock(nn.Module):
    """Non-local context block."""

    def __init__(self, in_channels: int, ratio: float = 1./4,
                 pooling_type: str = 'att', fusion_types: Sequence[str] = ('channel_add',),
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.channels = int(in_channels * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.LayerNorm([1, 1, 1]))
            self.attention = nn.Conv2d(in_channels, 1, kernel_size=1)

        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.channels, kernel_size=1),
                nn.LayerNorm([self.channels, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.channels, in_channels, kernel_size=1))

        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.channels, kernel_size=1),
                nn.LayerNorm([self.channels, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.channels, in_channels, kernel_size=1))

    def spatial_pool(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.pooling_type == 'att':
            attn = self.attention(x).view(B, 1, H * W)
            attn = torch.softmax(attn, dim=-1)
            context = torch.bmm(x.view(B, C, H * W), attn.transpose(1, 2))
            return context.view(B, C, 1, 1)
        else:
            return x.mean(dim=(2, 3), keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.spatial_pool(x)
        out = x
        if 'channel_mul' in self.fusion_types:
            channel_mul = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul
        if 'channel_add' in self.fusion_types:
            out = out + self.channel_add_conv(context)
        return out


# ---- NonLocal2d ----
class NonLocal2d(nn.Module):
    """Non-local self-attention block."""

    def __init__(self, in_channels: int, reduction: int = 2,
                 use_scale: bool = True, mode: str = 'embedded_gaussian',
                 sub_sample: bool = False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        inter_channels = in_channels // reduction
        self.inter_channels = inter_channels
        self.use_scale = use_scale
        self.mode = mode

        self.g = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.W = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels))
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        if mode == 'embedded_gaussian':
            self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
            self.phi = nn.Conv2d(in_channels, inter_channels, kernel_size=1)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(2))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        g = self.g(x).view(B, self.inter_channels, -1).permute(0, 2, 1)

        if self.mode == 'embedded_gaussian':
            theta = self.theta(x).view(B, self.inter_channels, -1).permute(0, 2, 1)
            phi = self.phi(x).view(B, self.inter_channels, -1)
            pairwise = torch.matmul(theta, phi)
            if self.use_scale:
                pairwise = pairwise / (self.inter_channels ** 0.5)
            pairwise = torch.softmax(pairwise, dim=-1)
        else:
            pairwise = torch.ones(B, H * W, H * W, device=x.device) / (H * W)

        y = torch.matmul(pairwise, g).permute(0, 2, 1).contiguous()
        y = y.view(B, self.inter_channels, H, W)
        z = self.W(y)
        return z + x


__all__ = [
    'ConvModule', 'DepthwiseSeparableConvModule', 'ContextBlock', 'NonLocal2d',
    'build_conv_layer', 'build_norm_layer', 'build_activation_layer',
    'build_upsample_layer', 'build_plugin_layer', 'build_dropout',
    'DropPath', 'Conv2dAdaptivePadding', 'Linear', 'Scale', 'Conv2d',
    'MultiheadAttention', 'FFN', 'BaseTransformerLayer',
]
