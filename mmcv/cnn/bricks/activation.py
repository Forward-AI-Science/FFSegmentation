from typing import Optional
import torch.nn as nn


def build_activation_layer(cfg: Optional[dict]) -> nn.Module:
    """Build activation layer from config dict."""
    if cfg is None:
        return nn.ReLU(inplace=True)
    cfg = cfg.copy()
    act_type = cfg.pop('type', 'ReLU')
    act_map = {
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'PReLU': nn.PReLU,
        'ReLU6': nn.ReLU6,
        'ELU': nn.ELU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'GELU': nn.GELU,
        'SiLU': nn.SiLU,
        'Swish': nn.SiLU,
        'Mish': nn.Mish,
        'Hardswish': nn.Hardswish,
        'Hardsigmoid': nn.Hardsigmoid,
        'SELU': nn.SELU,
        'CELU': nn.CELU,
        'Softplus': nn.Softplus,
        'Softshrink': nn.Softshrink,
        'Softsign': nn.Softsign,
        'Tanhshrink': nn.Tanhshrink,
        'Threshold': nn.Threshold,
        'Softmax': nn.Softmax,
        'LogSoftmax': nn.LogSoftmax,
        'Clip': nn.Hardtanh,
        'HSigmoid': nn.Hardsigmoid,
        'HSwish': nn.Hardswish,
    }
    cls = act_map.get(act_type)
    if cls is None:
        raise KeyError(f'Unknown activation type: {act_type}')
    return cls(**cfg)


__all__ = ['build_activation_layer']
