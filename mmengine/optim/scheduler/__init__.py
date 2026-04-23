from typing import Optional
import torch.optim as optim


class _BaseScheduler:
    """Common base for mmengine-style schedulers."""

    def __init__(self, optimizer, by_epoch: bool = True, begin: int = 0,
                 end: int = -1, last_step: int = -1):
        self.optimizer = optimizer
        self.by_epoch = by_epoch
        self.begin = begin
        self.end = end
        self._step_count = last_step + 1
        self._last_lr = [g['lr'] for g in optimizer.param_groups]

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        values = self.get_lr()
        for g, lr in zip(self.optimizer.param_groups, values):
            g['lr'] = lr
        self._last_lr = values
        self._step_count += 1

    def state_dict(self):
        return {'_step_count': self._step_count, '_last_lr': self._last_lr}

    def load_state_dict(self, state_dict):
        self._step_count = state_dict['_step_count']
        self._last_lr = state_dict['_last_lr']


class ConstantLR(_BaseScheduler):
    def __init__(self, optimizer, factor: float = 1./3, total_iters: int = 5,
                 **kwargs):
        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, **kwargs)
        self._base_lrs = [g['lr'] for g in optimizer.param_groups]

    def get_lr(self):
        if self._step_count < self.total_iters:
            return [lr * self.factor for lr in self._base_lrs]
        return list(self._base_lrs)


class LinearLR(_BaseScheduler):
    def __init__(self, optimizer, start_factor: float = 1./3,
                 end_factor: float = 1.0, total_iters: int = 5, **kwargs):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, **kwargs)
        self._base_lrs = [g['lr'] for g in optimizer.param_groups]

    def get_lr(self):
        t = min(self._step_count, self.total_iters)
        factor = (self.start_factor +
                  (self.end_factor - self.start_factor) * t / self.total_iters)
        return [lr * factor for lr in self._base_lrs]


class PolyLR(_BaseScheduler):
    """Polynomial learning rate decay."""

    def __init__(self, optimizer, eta_min: float = 0., power: float = 1.,
                 total_iters: int = 10, **kwargs):
        self.eta_min = eta_min
        self.power = power
        self.total_iters = total_iters
        super().__init__(optimizer, **kwargs)
        self._base_lrs = [g['lr'] for g in optimizer.param_groups]

    def get_lr(self):
        t = min(self._step_count, self.total_iters)
        return [
            self.eta_min + (base_lr - self.eta_min) *
            ((1 - t / self.total_iters) ** self.power)
            for base_lr in self._base_lrs
        ]


class lr_scheduler:
    """Namespace alias for scheduler classes."""
    ConstantLR = ConstantLR
    LinearLR = LinearLR
    PolyLR = PolyLR


__all__ = ['ConstantLR', 'LinearLR', 'PolyLR']
