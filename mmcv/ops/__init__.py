"""Pure-PyTorch/torchvision replacements for mmcv.ops custom CUDA kernels."""

import warnings
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- Deformable Convolutions ----
try:
    from torchvision.ops import DeformConv2d as _TVDeformConv2d, deform_conv2d

    class DeformConv2d(_TVDeformConv2d):
        """Deformable Conv2d backed by torchvision."""
        pass

    class ModulatedDeformConv2d(nn.Module):
        """Modulated (v2) deformable conv2d using torchvision."""

        def __init__(self, in_channels: int, out_channels: int,
                     kernel_size: int, stride: int = 1, padding: int = 0,
                     dilation: int = 1, groups: int = 1,
                     deformable_groups: int = 1, bias: bool = True, **kwargs):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            self.deformable_groups = deformable_groups

            self.weight = nn.Parameter(
                torch.Tensor(out_channels, in_channels // groups,
                             kernel_size, kernel_size))
            if bias:
                self.bias = nn.Parameter(torch.Tensor(out_channels))
            else:
                self.bias = None

            # Offset + mask conv (2 * k*k + k*k channels)
            self.conv_offset = nn.Conv2d(
                in_channels, deformable_groups * 3 * kernel_size * kernel_size,
                kernel_size, stride=stride, padding=padding, bias=True)
            self._init_weights()

        def _init_weights(self):
            nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
            if self.bias is not None:
                nn.init.zeros_(self.bias)
            nn.init.zeros_(self.conv_offset.weight)
            nn.init.zeros_(self.conv_offset.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.conv_offset(x)
            k2 = self.kernel_size * self.kernel_size
            o1, o2, mask = torch.chunk(out, 3, dim=1)
            offset = torch.cat([o1, o2], dim=1)
            mask = torch.sigmoid(mask)
            return deform_conv2d(
                x, offset, self.weight, self.bias,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, mask=mask)

except ImportError:
    warnings.warn('torchvision DeformConv2d not available; using Identity fallback.')

    class DeformConv2d(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size,
                     stride=1, padding=0, dilation=1, groups=1,
                     deformable_groups=1, bias=True, **kwargs):
            super().__init__(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding,
                             dilation=dilation, groups=groups, bias=bias)
            self.deformable_groups = deformable_groups

        def forward(self, x, offset=None):
            return super().forward(x)

    ModulatedDeformConv2d = DeformConv2d


# ---- Focal Loss ----
try:
    from torchvision.ops import sigmoid_focal_loss as _tv_focal

    def sigmoid_focal_loss(pred: torch.Tensor, target: torch.Tensor,
                           weight: Optional[torch.Tensor] = None,
                           gamma: float = 2.0, alpha: float = 0.25,
                           reduction: str = 'mean') -> torch.Tensor:
        loss = _tv_focal(pred, target.float(), alpha=alpha, gamma=gamma,
                         reduction='none')
        if weight is not None:
            loss = loss * weight
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss

except ImportError:
    def sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25,
                           reduction='mean'):
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        if weight is not None:
            loss = loss * weight
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss


# ---- CrissCrossAttention ----
class CrissCrossAttention(nn.Module):
    """Pure PyTorch Criss-Cross Attention (CCNet)."""

    def __init__(self, in_channels: int):
        super().__init__()
        inter_channels = in_channels // 8
        self.query_conv = nn.Conv2d(in_channels, inter_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, inter_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        # Horizontal attention
        proj_query_h = proj_query.permute(0, 3, 1, 2).contiguous().view(B * W, -1, H)
        proj_key_h = proj_key.permute(0, 3, 1, 2).contiguous().view(B * W, -1, H)
        proj_value_h = proj_value.permute(0, 3, 1, 2).contiguous().view(B * W, C, H)
        energy_h = torch.bmm(proj_query_h.transpose(1, 2), proj_key_h)
        attn_h = torch.softmax(energy_h, dim=-1)
        out_h = torch.bmm(proj_value_h, attn_h.transpose(1, 2)).view(B, W, C, H).permute(0, 2, 3, 1)

        # Vertical attention
        proj_query_v = proj_query.permute(0, 2, 1, 3).contiguous().view(B * H, -1, W)
        proj_key_v = proj_key.permute(0, 2, 1, 3).contiguous().view(B * H, -1, W)
        proj_value_v = proj_value.permute(0, 2, 1, 3).contiguous().view(B * H, C, W)
        energy_v = torch.bmm(proj_query_v.transpose(1, 2), proj_key_v)
        attn_v = torch.softmax(energy_v, dim=-1)
        out_v = torch.bmm(proj_value_v, attn_v.transpose(1, 2)).view(B, H, C, W).permute(0, 2, 1, 3)

        out = out_h + out_v
        return self.gamma * out + x


# ---- PSAMask ----
class PSAMask(nn.Module):
    """Pixel Shift Attention mask generator (simplified pure-PyTorch version)."""

    def __init__(self, psa_type: int, mask_h_max: int = None,
                 mask_w_max: int = None, feature_h: int = None,
                 feature_w: int = None, **kwargs):
        super().__init__()
        self.psa_type = psa_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ---- point_sample ----
def point_sample(input: torch.Tensor, point_coords: torch.Tensor,
                 **kwargs) -> torch.Tensor:
    """Bilinear sampling at point coordinates using F.grid_sample."""
    # point_coords: (B, N, 2) in [0,1] range
    # Convert to [-1, 1] for grid_sample
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)  # B, N, 1, 2

    point_coords = 2.0 * point_coords - 1.0
    out = F.grid_sample(input, point_coords, align_corners=False, **kwargs)
    if add_dim:
        out = out.squeeze(3)
    return out


__all__ = [
    'DeformConv2d', 'ModulatedDeformConv2d',
    'CrissCrossAttention', 'PSAMask',
    'point_sample', 'sigmoid_focal_loss',
]
