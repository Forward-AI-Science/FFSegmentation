import copy
import logging
from abc import ABCMeta
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .weight_init import (caffe2_xavier_init, constant_init, kaiming_init,
                           normal_init, trunc_normal_, trunc_normal_init,
                           xavier_init, _init_module)


class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base class for all modules, supports init_cfg weight initialization."""

    def __init__(self, init_cfg: Optional[Union[dict, List[dict]]] = None):
        super().__init__()
        self._is_init: bool = False
        self.init_cfg = copy.deepcopy(init_cfg)

    def init_weights(self):
        if self.init_cfg is None:
            return
        cfgs = self.init_cfg if isinstance(self.init_cfg, (list, tuple)) else [self.init_cfg]
        for cfg in cfgs:
            cfg = cfg.copy()
            init_type = cfg.get('type', '')
            if init_type == 'Pretrained':
                checkpoint = cfg.get('checkpoint', '')
                prefix = cfg.get('prefix', None)
                if checkpoint:
                    try:
                        from mmengine.runner.checkpoint import load_checkpoint
                        load_checkpoint(self, checkpoint, map_location='cpu',
                                        prefix=prefix, logger=None)
                    except Exception as e:
                        import warnings
                        warnings.warn(f'Failed to load pretrained: {e}')
            else:
                _init_module(self, cfg)
        self._is_init = True

    def __repr__(self) -> str:
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s


class BaseModel(BaseModule):
    """Base class for all models in mmengine/mmseg style.

    Wraps nn.Module with train_step / val_step / test_step plus a
    data_preprocessor that moves data to the right device.
    """

    def __init__(self, data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[Union[dict, List[dict]]] = None):
        super().__init__(init_cfg=init_cfg)
        if data_preprocessor is None:
            self.data_preprocessor = BaseDataPreprocessor()
        elif isinstance(data_preprocessor, dict):
            from mmengine.registry import MODELS
            self.data_preprocessor = MODELS.build(data_preprocessor)
        else:
            self.data_preprocessor = data_preprocessor

    def forward(self, inputs, data_samples=None, mode: str = 'tensor'):
        raise NotImplementedError

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper) -> Dict[str, torch.Tensor]:
        data = self.data_preprocessor(data, training=True)
        losses = self(**data, mode='loss')
        parsed = self.parse_losses(losses)
        optim_wrapper.update_params(parsed[0])
        return parsed[1]

    def val_step(self, data: Union[dict, tuple, list]) -> list:
        data = self.data_preprocessor(data, training=False)
        return self(**data, mode='predict')

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        data = self.data_preprocessor(data, training=False)
        return self(**data, mode='predict')

    @staticmethod
    def parse_losses(losses: Dict[str, torch.Tensor]):
        log_vars: Dict[str, float] = {}
        total_loss = sum(v for v in losses.values() if isinstance(v, torch.Tensor))
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                log_vars[k] = v.mean().item()
        return total_loss, log_vars


class BaseDataPreprocessor(BaseModule):
    """Simple data preprocessor: stacks inputs and moves to device."""

    def __init__(self, mean=None, std=None, bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False, non_blocking: bool = False,
                 init_cfg=None, **kwargs):
        super().__init__(init_cfg=init_cfg)
        self._mean_val = mean
        self._std_val = std
        self.bgr_to_rgb = bgr_to_rgb
        self.rgb_to_bgr = rgb_to_bgr
        self.non_blocking = non_blocking
        # Don't pre-register mean/std here — subclasses do it themselves

    def forward(self, data: Union[dict, tuple, list], training: bool = False):
        if isinstance(data, dict):
            inputs = data.get('inputs', None)
            data_samples = data.get('data_samples', None)
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            inputs, data_samples = data
        else:
            inputs, data_samples = data, None

        inputs = self._preprocess_inputs(inputs)
        return {'inputs': inputs, 'data_samples': data_samples}

    def _preprocess_inputs(self, inputs):
        if inputs is None:
            return inputs
        if isinstance(inputs, (list, tuple)):
            if all(isinstance(x, torch.Tensor) for x in inputs):
                try:
                    inputs = torch.stack(inputs, dim=0)
                except RuntimeError:
                    pass
            else:
                return inputs
        if isinstance(inputs, torch.Tensor):
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = inputs.device
            inputs = inputs.to(device, non_blocking=self.non_blocking)
            inputs = inputs.float()
            if self.bgr_to_rgb:
                inputs = inputs[:, [2, 1, 0]]
            elif self.rgb_to_bgr:
                inputs = inputs[:, [2, 1, 0]]
            mean = getattr(self, 'mean', None)
            std = getattr(self, 'std', None)
            if mean is not None and std is not None:
                inputs = (inputs - mean) / std
        return inputs

    def _cast_data(self, data, device=None):
        if device is None:
            device = next(self.parameters(), torch.tensor(0)).device
        if isinstance(data, torch.Tensor):
            return data.to(device, non_blocking=self.non_blocking)
        elif isinstance(data, dict):
            return {k: self._cast_data(v, device) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self._cast_data(x, device) for x in data)
        return data

    def cast_data(self, data):
        """Public alias for _cast_data used by subclasses."""
        try:
            device = next(self.parameters()).device
        except StopIteration:
            try:
                device = next(self.buffers()).device
            except StopIteration:
                device = torch.device('cpu')
        return self._cast_data(data, device)


class BaseTTAModel(BaseModule):
    """Base class for test-time augmentation model wrapper."""

    def __init__(self, model: nn.Module, tta_cfg: Optional[dict] = None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.module = model
        self.tta_cfg = tta_cfg or {}

    def forward(self, inputs, data_samples=None, mode='predict'):
        assert mode == 'predict'
        return self._aug_test(inputs, data_samples)

    def _aug_test(self, inputs, data_samples):
        return self.module(inputs, data_samples, mode='predict')

    def test_step(self, data):
        data = self.module.data_preprocessor(data, training=False)
        return self(**data, mode='predict')


# ---- nn.Module wrappers ----

class ModuleList(nn.ModuleList):
    """nn.ModuleList with init_cfg support."""
    def __init__(self, modules=None, init_cfg=None):
        super().__init__(modules)
        self.init_cfg = init_cfg


class Sequential(nn.Sequential):
    """nn.Sequential with init_cfg support."""
    def __init__(self, *args, init_cfg=None):
        super().__init__(*args)
        self.init_cfg = init_cfg


def revert_sync_batchnorm(module: nn.Module) -> nn.Module:
    """Convert all SyncBatchNorm to BatchNorm2d."""
    module_output = module
    if isinstance(module, nn.SyncBatchNorm):
        module_output = nn.BatchNorm2d(
            module.num_features, module.eps, module.momentum,
            module.affine, module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight.copy_(module.weight)
                module_output.bias.copy_(module.bias)
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    return module_output


__all__ = [
    'BaseModule', 'BaseModel', 'BaseDataPreprocessor', 'BaseTTAModel',
    'ModuleList', 'Sequential', 'revert_sync_batchnorm',
    'constant_init', 'xavier_init', 'normal_init', 'trunc_normal_',
    'trunc_normal_init', 'kaiming_init', 'caffe2_xavier_init',
]
