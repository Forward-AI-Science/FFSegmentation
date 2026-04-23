import logging
import os
import os.path as osp
import time
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoint import (CheckpointLoader, _load_checkpoint,
                         _load_checkpoint_to_model, load_checkpoint,
                         load_state_dict)
from .loops import (EpochBasedTrainLoop, IterBasedTrainLoop, TestLoop, ValLoop)


class Runner:
    """Minimal training runner compatible with mmengine.runner.Runner."""

    def __init__(self, model, work_dir: str = 'work_dir',
                 train_dataloader=None, val_dataloader=None,
                 test_dataloader=None, train_cfg: Optional[dict] = None,
                 val_cfg: Optional[dict] = None, test_cfg: Optional[dict] = None,
                 optim_wrapper: Optional[dict] = None,
                 param_scheduler=None,
                 val_evaluator=None, test_evaluator=None,
                 default_hooks: Optional[dict] = None,
                 custom_hooks: Optional[list] = None,
                 log_level: int = logging.INFO,
                 resume: bool = False,
                 load_from: Optional[str] = None,
                 launcher: str = 'none',
                 env_cfg: Optional[dict] = None,
                 visualizer=None,
                 log_processor: Optional[dict] = None,
                 default_scope: Optional[str] = None,
                 cfg=None, **kwargs):

        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

        # Logger
        from mmengine.logging import MMLogger
        self.logger = MMLogger.get_instance(
            'runner',
            log_file=osp.join(work_dir, 'run.log'),
            log_level=log_level)

        # Model
        self.model = model

        # Device
        self.device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')

        # Dataloaders (may be dicts or DataLoader objects)
        self.train_dataloader = self._build_dataloader(train_dataloader)
        self.val_dataloader = self._build_dataloader(val_dataloader)
        self.test_dataloader = self._build_dataloader(test_dataloader)

        # Optimizer
        self.optim_wrapper = self._build_optim_wrapper(optim_wrapper)

        # Schedulers
        self.param_schedulers = self._build_schedulers(param_scheduler)

        # Training mode
        self.train_cfg = train_cfg or {}
        self._by_epoch = 'max_epochs' in self.train_cfg
        self.max_epochs = self.train_cfg.get('max_epochs', 0)
        self.max_iters = self.train_cfg.get('max_iters', 0)
        if self._by_epoch and self.max_iters == 0 and self.train_dataloader:
            self.max_iters = self.max_epochs * len(self.train_dataloader)

        self.val_interval = self.train_cfg.get('val_interval', 1)
        self.val_begin = self.train_cfg.get('val_begin', 1)

        # Evaluators
        self.val_evaluator = self._build_evaluator(val_evaluator)
        self.test_evaluator = self._build_evaluator(test_evaluator)

        # Hooks
        self.hooks = self._build_hooks(default_hooks, custom_hooks)

        # State
        self.epoch = 0
        self.iter = 0
        self.resume = resume
        self.load_from = load_from

        if load_from:
            load_checkpoint(self.model, load_from, map_location='cpu')
        if resume:
            self._resume()

    @classmethod
    def from_cfg(cls, cfg) -> 'Runner':
        """Build Runner from a Config object."""
        cfg_dict = dict(cfg.items()) if hasattr(cfg, 'items') else {}
        return cls._from_cfg_dict(cfg_dict)

    @classmethod
    def _from_cfg_dict(cls, cfg: dict) -> 'Runner':
        from mmengine.registry import MODELS, DATASETS, METRICS, TRANSFORMS
        from torch.utils.data import DataLoader

        # Build model
        model_cfg = cfg.get('model', {})
        model = MODELS.build(model_cfg)

        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        kwargs = dict(
            model=model,
            work_dir=cfg.get('work_dir', 'work_dir'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            resume=cfg.get('resume', False),
            load_from=cfg.get('load_from'),
            launcher=cfg.get('launcher', 'none'),
            log_level=logging.INFO,
        )

        # Build dataloaders
        for split in ('train', 'val', 'test'):
            dl_cfg = cfg.get(f'{split}_dataloader')
            kwargs[f'{split}_dataloader'] = dl_cfg

        # Build evaluators
        for split in ('val', 'test'):
            ev_cfg = cfg.get(f'{split}_evaluator')
            kwargs[f'{split}_evaluator'] = ev_cfg

        return cls(**kwargs)

    def _build_dataloader(self, cfg):
        if cfg is None or isinstance(cfg, DataLoader):
            return cfg
        if not isinstance(cfg, dict):
            return cfg
        from mmengine.registry import DATASETS, TRANSFORMS
        from mmengine.dataset import DefaultSampler

        cfg = cfg.copy()
        dataset_cfg = cfg.pop('dataset', None)
        if dataset_cfg is None:
            return None
        dataset = DATASETS.build(dataset_cfg)

        batch_size = cfg.pop('batch_size', 1)
        num_workers = cfg.pop('num_workers', 0)
        sampler_cfg = cfg.pop('sampler', {'type': 'DefaultSampler', 'shuffle': True})
        if isinstance(sampler_cfg, dict):
            sc = sampler_cfg.copy()
            sc.pop('type', None)
            shuffle = sc.pop('shuffle', True)
            sampler = DefaultSampler(dataset, shuffle=shuffle)
        else:
            sampler = sampler_cfg

        persistent_workers = cfg.pop('persistent_workers', False)
        pin_memory = cfg.pop('pin_memory', False)
        prefetch_factor = cfg.pop('prefetch_factor', None)
        drop_last = cfg.pop('drop_last', False)
        collate_fn = cfg.pop('collate_fn', None)

        dl_kwargs = dict(
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        if num_workers > 0:
            dl_kwargs['persistent_workers'] = persistent_workers
            dl_kwargs['pin_memory'] = pin_memory
        if collate_fn is not None:
            dl_kwargs['collate_fn'] = collate_fn

        return DataLoader(dataset, **dl_kwargs)

    def _build_optim_wrapper(self, cfg):
        if cfg is None:
            return None
        if not isinstance(cfg, dict):
            return cfg
        from mmengine.optim import OptimWrapper, AmpOptimWrapper, DefaultOptimWrapperConstructor
        cfg = cfg.copy()
        wrapper_type = cfg.pop('type', 'OptimWrapper')
        optimizer_cfg = cfg.pop('optimizer', {}).copy()
        opt_type = optimizer_cfg.pop('type', 'SGD')
        lr = optimizer_cfg.pop('lr', 0.01)
        opt_cls = getattr(torch.optim, opt_type, None)
        if opt_cls is None:
            raise ValueError(f'Unknown optimizer: {opt_type}')
        optimizer = opt_cls(self.model.parameters(), lr=lr, **optimizer_cfg)
        if wrapper_type == 'AmpOptimWrapper':
            return AmpOptimWrapper(optimizer, **cfg)
        return OptimWrapper(optimizer, **cfg)

    def _build_schedulers(self, cfg):
        if cfg is None:
            return []
        if not isinstance(cfg, (list, tuple)):
            cfg = [cfg]
        schedulers = []
        for sc in cfg:
            if isinstance(sc, dict):
                sc = sc.copy()
                sc_type = sc.pop('type', 'PolyLR')
                from mmengine.optim.scheduler import PolyLR, LinearLR, ConstantLR
                cls_map = {'PolyLR': PolyLR, 'LinearLR': LinearLR, 'ConstantLR': ConstantLR}
                cls = cls_map.get(sc_type)
                if cls and self.optim_wrapper:
                    schedulers.append(cls(self.optim_wrapper.optimizer, **sc))
        return schedulers

    def _build_evaluator(self, cfg):
        if cfg is None:
            return None
        if not isinstance(cfg, dict):
            return cfg
        from mmengine.registry import METRICS
        return METRICS.build(cfg)

    def _build_hooks(self, default_hooks, custom_hooks):
        from mmengine.hooks import (CheckpointHook, IterTimerHook,
                                    DistSamplerSeedHook, LoggerHook,
                                    ParamSchedulerHook)
        hooks = []
        dh = default_hooks or {}
        if 'checkpoint' in dh:
            hooks.append(CheckpointHook(out_dir=self.work_dir,
                                        **{k: v for k, v in dh['checkpoint'].items()
                                           if k != 'type'}))
        else:
            hooks.append(CheckpointHook(out_dir=self.work_dir))
        hooks.append(IterTimerHook())
        hooks.append(DistSamplerSeedHook())
        hooks.append(LoggerHook())
        hooks.append(ParamSchedulerHook())
        for h in (custom_hooks or []):
            if isinstance(h, dict):
                from mmengine.registry import HOOKS
                hooks.append(HOOKS.build(h))
            else:
                hooks.append(h)
        return hooks

    def _resume(self):
        ckpt_path = osp.join(self.work_dir, 'last.pth')
        if osp.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(ckpt.get('state_dict', ckpt))
            self.epoch = ckpt.get('epoch', 0)
            self.iter = ckpt.get('iter', 0)
            if 'optimizer' in ckpt and self.optim_wrapper:
                self.optim_wrapper.optimizer.load_state_dict(ckpt['optimizer'])
            self.logger.info(f'Resumed from {ckpt_path} (epoch {self.epoch})')

    def call_hook(self, fn_name: str, **kwargs):
        for hook in self.hooks:
            fn = getattr(hook, fn_name, None)
            if fn is not None:
                fn(self, **kwargs)

    def train(self):
        self.model.train()
        self.call_hook('before_run')
        self.call_hook('before_train')

        if self._by_epoch:
            self._train_by_epoch()
        else:
            self._train_by_iter()

        self.call_hook('after_train')
        self.call_hook('after_run')

    def _train_by_epoch(self):
        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            self.call_hook('before_train_epoch')
            for batch_idx, data_batch in enumerate(self.train_dataloader):
                self.call_hook('before_train_iter', batch_idx=batch_idx,
                               data_batch=data_batch)
                self.model.train()
                outputs = self.model.train_step(data_batch, self.optim_wrapper)
                self.iter += 1
                self.call_hook('after_train_iter', batch_idx=batch_idx,
                               data_batch=data_batch, outputs=outputs)
            self.call_hook('after_train_epoch')
            # Validation
            if (epoch + 1 >= self.val_begin and
                    (epoch + 1) % self.val_interval == 0):
                self.val()

    def _train_by_iter(self):
        iter_count = self.iter
        data_iter = iter(self.train_dataloader)
        self.call_hook('before_train_epoch')
        while iter_count < self.max_iters:
            try:
                data_batch = next(data_iter)
            except StopIteration:
                self.call_hook('after_train_epoch')
                self.epoch += 1
                self.call_hook('before_train_epoch')
                data_iter = iter(self.train_dataloader)
                data_batch = next(data_iter)

            self.call_hook('before_train_iter', batch_idx=iter_count,
                           data_batch=data_batch)
            self.model.train()
            outputs = self.model.train_step(data_batch, self.optim_wrapper)
            self.iter = iter_count
            iter_count += 1
            self.call_hook('after_train_iter', batch_idx=iter_count,
                           data_batch=data_batch, outputs=outputs)

            if iter_count % self.val_interval == 0 and iter_count >= self.val_begin:
                self.val()

    @torch.no_grad()
    def val(self):
        if self.val_dataloader is None:
            return
        self.model.eval()
        self.call_hook('before_val')
        self.call_hook('before_val_epoch')
        results = []
        for batch_idx, data_batch in enumerate(self.val_dataloader):
            self.call_hook('before_val_iter', batch_idx=batch_idx,
                           data_batch=data_batch)
            outputs = self.model.val_step(data_batch)
            results.extend(outputs)
            self.call_hook('after_val_iter', batch_idx=batch_idx,
                           data_batch=data_batch, outputs=outputs)
        metrics = {}
        if self.val_evaluator:
            metrics = self.val_evaluator.evaluate(len(results))
        self.call_hook('after_val_epoch', metrics=metrics)
        self.call_hook('after_val')
        return metrics

    @torch.no_grad()
    def test(self):
        if self.test_dataloader is None:
            return
        self.model.eval()
        self.call_hook('before_test')
        self.call_hook('before_test_epoch')
        results = []
        for batch_idx, data_batch in enumerate(self.test_dataloader):
            self.call_hook('before_test_iter', batch_idx=batch_idx,
                           data_batch=data_batch)
            outputs = self.model.test_step(data_batch)
            if self.test_evaluator:
                self.test_evaluator.process(data_batch, outputs)
            results.extend(outputs)
            self.call_hook('after_test_iter', batch_idx=batch_idx,
                           data_batch=data_batch, outputs=outputs)
        metrics = {}
        if self.test_evaluator:
            metrics = self.test_evaluator.evaluate(len(results))
        self.call_hook('after_test_epoch', metrics=metrics)
        self.call_hook('after_test')
        self.logger.info(f'Test results: {metrics}')
        return metrics


__all__ = [
    'Runner', 'CheckpointLoader', '_load_checkpoint', 'load_state_dict',
    '_load_checkpoint_to_model', 'load_checkpoint',
    'IterBasedTrainLoop', 'EpochBasedTrainLoop', 'ValLoop', 'TestLoop',
]
