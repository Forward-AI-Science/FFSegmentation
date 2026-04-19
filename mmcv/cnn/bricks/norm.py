from typing import Dict, Optional, Tuple, Union
import torch.nn as nn


def build_norm_layer(cfg: Optional[dict], num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """Build normalization layer from config.

    Returns:
        (name, layer): name is the attribute name, layer is the nn.Module.
    """
    if cfg is None:
        return '', nn.Identity()
    cfg = cfg.copy()
    norm_type = cfg.pop('type', 'BN')
    requires_grad = cfg.pop('requires_grad', True)
    postfix = str(postfix)

    norm_map = {
        'BN': nn.BatchNorm2d,
        'BN1d': nn.BatchNorm1d,
        'BN2d': nn.BatchNorm2d,
        'BN3d': nn.BatchNorm3d,
        'SyncBN': nn.SyncBatchNorm,
        'GN': nn.GroupNorm,
        'LN': nn.LayerNorm,
        'IN': nn.InstanceNorm2d,
        'IN1d': nn.InstanceNorm1d,
        'IN2d': nn.InstanceNorm2d,
        'IN3d': nn.InstanceNorm3d,
    }

    cls = norm_map.get(norm_type)
    if cls is None:
        raise KeyError(f'Unknown norm type: {norm_type}')

    if norm_type == 'GN':
        num_groups = cfg.pop('num_groups', 32)
        layer = cls(num_groups, num_features, **cfg)
    elif norm_type == 'LN':
        layer = cls(num_features, **cfg)
    else:
        layer = cls(num_features, **cfg)

    for p in layer.parameters():
        p.requires_grad = requires_grad

    abbr_map = {'BN': 'bn', 'BN1d': 'bn', 'BN2d': 'bn', 'BN3d': 'bn',
                'SyncBN': 'bn', 'GN': 'gn', 'LN': 'ln',
                'IN': 'in', 'IN1d': 'in', 'IN2d': 'in', 'IN3d': 'in'}
    abbr = abbr_map.get(norm_type, 'norm')
    name = abbr + postfix
    return name, layer


__all__ = ['build_norm_layer']
