import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic Depth (DropPath) regularization."""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor

    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob:.3f}'


def build_dropout(cfg: dict, default_args: dict = None) -> nn.Module:
    if cfg is None:
        return nn.Identity()
    cfg = cfg.copy()
    dropout_type = cfg.pop('type', 'Dropout')
    if default_args:
        cfg.update({k: v for k, v in default_args.items() if k not in cfg})
    if dropout_type == 'DropPath':
        return DropPath(**cfg)
    elif dropout_type == 'Dropout':
        return nn.Dropout(**cfg)
    elif dropout_type == 'Dropout2d':
        return nn.Dropout2d(**cfg)
    else:
        raise ValueError(f'Unknown dropout type: {dropout_type}')


__all__ = ['DropPath', 'build_dropout']
