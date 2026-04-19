from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler


class OptimWrapper:
    """Optimizer wrapper that mimics mmengine.optim.OptimWrapper."""

    def __init__(self, optimizer: torch.optim.Optimizer,
                 clip_grad: Optional[dict] = None,
                 accumulative_counts: int = 1):
        self.optimizer = optimizer
        self.clip_grad = clip_grad
        self.accumulative_counts = accumulative_counts
        self._inner_count = 0

    def update_params(self, loss: torch.Tensor, step_kwargs: Optional[dict] = None):
        self._inner_count += 1
        scaled_loss = loss / self.accumulative_counts
        scaled_loss.backward()
        if self._inner_count % self.accumulative_counts == 0:
            if self.clip_grad:
                params = [p for g in self.optimizer.param_groups
                          for p in g['params'] if p.grad is not None]
                torch.nn.utils.clip_grad_norm_(
                    params, self.clip_grad.get('max_norm', 1.0))
            self.optimizer.step()
            self.optimizer.zero_grad()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.optimizer.load_state_dict(state_dict)

    @contextmanager
    def optim_context(self, model: nn.Module) -> Iterator:
        yield

    @property
    def param_groups(self):
        return self.optimizer.param_groups


class AmpOptimWrapper(OptimWrapper):
    """AMP-enabled optimizer wrapper."""

    def __init__(self, optimizer: torch.optim.Optimizer,
                 loss_scale: Union[str, float] = 512.,
                 clip_grad: Optional[dict] = None,
                 accumulative_counts: int = 1, **kwargs):
        super().__init__(optimizer, clip_grad, accumulative_counts)
        if loss_scale == 'dynamic':
            self.scaler = GradScaler()
        else:
            self.scaler = GradScaler(init_scale=float(loss_scale))

    def update_params(self, loss: torch.Tensor, step_kwargs=None):
        self._inner_count += 1
        scaled_loss = loss / self.accumulative_counts
        self.scaler.scale(scaled_loss).backward()
        if self._inner_count % self.accumulative_counts == 0:
            if self.clip_grad:
                self.scaler.unscale_(self.optimizer)
                params = [p for g in self.optimizer.param_groups
                          for p in g['params'] if p.grad is not None]
                torch.nn.utils.clip_grad_norm_(
                    params, self.clip_grad.get('max_norm', 1.0))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

    @contextmanager
    def optim_context(self, model: nn.Module) -> Iterator:
        from torch.cuda.amp import autocast
        with autocast():
            yield


class DefaultOptimWrapperConstructor:
    """Build OptimWrapper from config dict."""

    def __init__(self, optim_wrapper_cfg: dict,
                 paramwise_cfg: Optional[dict] = None):
        self.optim_wrapper_cfg = optim_wrapper_cfg
        self.paramwise_cfg = paramwise_cfg

    def __call__(self, model: nn.Module) -> OptimWrapper:
        cfg = self.optim_wrapper_cfg.copy()
        wrapper_type = cfg.pop('type', 'OptimWrapper')
        optimizer_cfg = cfg.pop('optimizer', {}).copy()
        opt_type = optimizer_cfg.pop('type', 'SGD')
        lr = optimizer_cfg.pop('lr', 0.01)

        # Get optimizer class
        opt_cls = getattr(torch.optim, opt_type, None)
        if opt_cls is None:
            raise ValueError(f'Unknown optimizer: {opt_type}')

        optimizer = opt_cls(model.parameters(), lr=lr, **optimizer_cfg)

        if wrapper_type == 'AmpOptimWrapper':
            return AmpOptimWrapper(optimizer, **cfg)
        return OptimWrapper(optimizer, **cfg)


__all__ = ['OptimWrapper', 'AmpOptimWrapper', 'DefaultOptimWrapperConstructor']
