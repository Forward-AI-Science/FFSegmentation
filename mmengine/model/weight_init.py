import math
import warnings
from typing import Optional, Union

import torch
import torch.nn as nn


def constant_init(module: nn.Module, val: float, bias: float = 0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module: nn.Module, gain: float = 1,
                bias: float = 0, distribution: str = 'normal'):
    assert distribution in ('uniform', 'normal')
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module: nn.Module, mean: float = 0,
                std: float = 1, bias: float = 0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def trunc_normal_(tensor: torch.Tensor, mean: float = 0., std: float = 1.,
                  a: float = -2., b: float = 2.) -> torch.Tensor:
    # Truncated normal using the erfinv method
    with torch.no_grad():
        l_val = (1. + math.erf((a - mean) / (std * math.sqrt(2.)))) / 2.
        u_val = (1. + math.erf((b - mean) / (std * math.sqrt(2.)))) / 2.
        tensor.uniform_(l_val, u_val)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(a, b)
    return tensor


def kaiming_init(module: nn.Module, a: float = 0, mode: str = 'fan_out',
                 nonlinearity: str = 'relu', bias: float = 0,
                 distribution: str = 'normal'):
    assert distribution in ('uniform', 'normal')
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode,
                                     nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode,
                                    nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def caffe2_xavier_init(module: nn.Module, bias: float = 0):
    # Caffe2 uses fan_in xavier uniform
    kaiming_init(module, a=1, mode='fan_in', nonlinearity='leaky_relu',
                 distribution='uniform', bias=bias)


def trunc_normal_init(module: nn.Module, mean: float = 0., std: float = 1.,
                      a: float = -2., b: float = 2., bias: float = 0.):
    """Initialize module weights with truncated normal distribution."""
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean=mean, std=std, a=a, b=b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def update_init_info(module: nn.Module, init_info: str):
    if not hasattr(module, '_params_init_info'):
        module._params_init_info = {}
    for name, param in module.named_parameters():
        if name not in module._params_init_info:
            module._params_init_info[name] = dict(
                init_info=init_info, tmp_mean_value=param.data.mean().item())


def _init_module(module: nn.Module, init_cfg: dict):
    init_type = init_cfg.get('type', '')
    layer_names = init_cfg.get('layer', None)
    override = init_cfg.get('override', None)

    def _matches(m):
        if layer_names is None:
            return True
        cls_names = [layer_names] if isinstance(layer_names, str) else layer_names
        return any(type(m).__name__ == n for n in cls_names)

    for m in module.modules():
        if not _matches(m):
            continue
        if init_type == 'Constant':
            constant_init(m, val=init_cfg.get('val', 0),
                          bias=init_cfg.get('bias', 0))
        elif init_type == 'Xavier':
            xavier_init(m, gain=init_cfg.get('gain', 1),
                        bias=init_cfg.get('bias', 0),
                        distribution=init_cfg.get('distribution', 'normal'))
        elif init_type == 'Normal':
            normal_init(m, mean=init_cfg.get('mean', 0),
                        std=init_cfg.get('std', 1),
                        bias=init_cfg.get('bias', 0))
        elif init_type == 'Kaiming':
            kaiming_init(m, a=init_cfg.get('a', 0),
                         mode=init_cfg.get('mode', 'fan_out'),
                         nonlinearity=init_cfg.get('nonlinearity', 'relu'),
                         bias=init_cfg.get('bias', 0),
                         distribution=init_cfg.get('distribution', 'normal'))
        elif init_type == 'TruncNormal':
            if hasattr(m, 'weight') and m.weight is not None:
                trunc_normal_(m.weight, mean=init_cfg.get('mean', 0.),
                               std=init_cfg.get('std', 1.),
                               a=init_cfg.get('a', -2.),
                               b=init_cfg.get('b', 2.))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, init_cfg.get('bias', 0))
        elif init_type == 'Pretrained':
            pass  # handled in BaseModule.init_weights


__all__ = [
    'constant_init', 'xavier_init', 'normal_init', 'trunc_normal_',
    'kaiming_init', 'caffe2_xavier_init', 'update_init_info', '_init_module',
]
