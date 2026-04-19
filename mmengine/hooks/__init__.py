import logging
import os
import os.path as osp
from typing import Any, Dict, List, Optional, Sequence, Union


class Hook:
    """Base class for all hooks in training."""

    priority = 'NORMAL'

    def before_run(self, runner) -> None:
        pass

    def after_run(self, runner) -> None:
        pass

    def before_train(self, runner) -> None:
        pass

    def after_train(self, runner) -> None:
        pass

    def before_train_epoch(self, runner) -> None:
        pass

    def after_train_epoch(self, runner, metrics: Optional[dict] = None) -> None:
        pass

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        pass

    def after_train_iter(self, runner, batch_idx: int, data_batch=None,
                         outputs: Optional[dict] = None) -> None:
        pass

    def before_val(self, runner) -> None:
        pass

    def after_val(self, runner) -> None:
        pass

    def before_val_epoch(self, runner) -> None:
        pass

    def after_val_epoch(self, runner, metrics: Optional[dict] = None) -> None:
        pass

    def before_val_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        pass

    def after_val_iter(self, runner, batch_idx: int, data_batch=None,
                       outputs: Optional[Sequence] = None) -> None:
        pass

    def before_test(self, runner) -> None:
        pass

    def after_test(self, runner) -> None:
        pass

    def before_test_epoch(self, runner) -> None:
        pass

    def after_test_epoch(self, runner, metrics: Optional[dict] = None) -> None:
        pass

    def before_test_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        pass

    def after_test_iter(self, runner, batch_idx: int, data_batch=None,
                        outputs: Optional[Sequence] = None) -> None:
        pass

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        pass

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        pass

    def every_n_epochs(self, runner, n: int) -> bool:
        return runner.epoch % n == 0

    def every_n_iters(self, runner, n: int) -> bool:
        return runner.iter % n == 0

    def end_of_epoch(self, dataloader, batch_idx: int) -> bool:
        return batch_idx + 1 == len(dataloader)

    def is_last_train_epoch(self, runner) -> bool:
        return runner.epoch + 1 == runner.max_epochs

    def is_last_train_iter(self, runner) -> bool:
        return runner.iter + 1 == runner.max_iters


class CheckpointHook(Hook):
    priority = 'VERY_LOW'

    def __init__(self, interval: int = 1, by_epoch: bool = True,
                 save_optimizer: bool = True, out_dir: Optional[str] = None,
                 max_keep_ckpts: int = -1, save_last: bool = True,
                 save_best: Optional[str] = None, rule: Optional[str] = None,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.save_best = save_best
        self.rule = rule
        self._best_metric = None
        self._saved_ckpts: List[str] = []

    def before_run(self, runner) -> None:
        if self.out_dir is None:
            self.out_dir = runner.work_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def after_train_epoch(self, runner, metrics=None) -> None:
        if self.by_epoch and self.every_n_epochs(runner, self.interval):
            self._save(runner, f'epoch_{runner.epoch}.pth')

    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None) -> None:
        if not self.by_epoch and self.every_n_iters(runner, self.interval):
            self._save(runner, f'iter_{runner.iter}.pth')

    def after_val_epoch(self, runner, metrics=None) -> None:
        if self.save_best and metrics:
            metric_val = metrics.get(self.save_best)
            if metric_val is not None:
                rule = self.rule or 'greater'
                is_better = (self._best_metric is None or
                             (rule == 'greater' and metric_val > self._best_metric) or
                             (rule == 'less' and metric_val < self._best_metric))
                if is_better:
                    self._best_metric = metric_val
                    self._save(runner, 'best.pth')

    def _save(self, runner, filename: str):
        import torch
        filepath = osp.join(self.out_dir, filename)
        ckpt = {'epoch': runner.epoch, 'iter': runner.iter,
                'state_dict': runner.model.state_dict()}
        if self.save_optimizer and hasattr(runner, 'optim_wrapper'):
            ckpt['optimizer'] = runner.optim_wrapper.optimizer.state_dict()
        torch.save(ckpt, filepath)
        self._saved_ckpts.append(filepath)
        if self.max_keep_ckpts > 0 and len(self._saved_ckpts) > self.max_keep_ckpts:
            to_remove = self._saved_ckpts.pop(0)
            if os.path.isfile(to_remove):
                os.remove(to_remove)
        if self.save_last:
            last_path = osp.join(self.out_dir, 'last.pth')
            torch.save(ckpt, last_path)


class IterTimerHook(Hook):
    priority = 'NORMAL'

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        import time
        self._start = time.perf_counter()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        import time
        elapsed = time.perf_counter() - self._start


class DistSamplerSeedHook(Hook):
    priority = 'NORMAL'

    def before_train_epoch(self, runner):
        dl = runner.train_dataloader
        if hasattr(dl, 'sampler') and hasattr(dl.sampler, 'set_epoch'):
            dl.sampler.set_epoch(runner.epoch)


class LoggerHook(Hook):
    priority = 'BELOW_NORMAL'

    def __init__(self, interval: int = 10, by_epoch: bool = True, **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if outputs and self.every_n_iters(runner, self.interval):
            msg = f'Epoch [{runner.epoch}][{batch_idx}/{len(runner.train_dataloader)}] '
            if isinstance(outputs, dict):
                for k, v in outputs.items():
                    msg += f'{k}: {v:.4f} '
            runner.logger.info(msg)


class ParamSchedulerHook(Hook):
    priority = 'LOW'

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        for scheduler in getattr(runner, 'param_schedulers', []):
            if not getattr(scheduler, 'by_epoch', True):
                scheduler.step()

    def after_train_epoch(self, runner, metrics=None):
        for scheduler in getattr(runner, 'param_schedulers', []):
            if getattr(scheduler, 'by_epoch', True):
                scheduler.step()


__all__ = [
    'Hook', 'CheckpointHook', 'IterTimerHook', 'DistSamplerSeedHook',
    'LoggerHook', 'ParamSchedulerHook',
]
