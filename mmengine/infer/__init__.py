from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn


ModelType = Union[dict, str, nn.Module]


class BaseInferencer(metaclass=ABCMeta):
    """Base class for inferencers."""

    def __init__(self, model: ModelType = None,
                 weights: Optional[str] = None,
                 device: Optional[str] = None, **kwargs):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        if isinstance(model, nn.Module):
            self.model = model.to(self.device)
        elif isinstance(model, dict):
            from mmengine.registry import MODELS
            self.model = MODELS.build(model).to(self.device)
        else:
            self.model = model
        if weights and self.model is not None:
            from mmengine.runner import load_checkpoint
            load_checkpoint(self.model, weights, map_location=device)
        if self.model is not None:
            self.model.eval()

    def __call__(self, inputs, **kwargs):
        return self.forward(inputs, **kwargs)

    @abstractmethod
    def forward(self, inputs, **kwargs):
        pass

    def preprocess(self, inputs, **kwargs):
        return inputs

    def postprocess(self, preds, **kwargs):
        return preds


class infer:
    BaseInferencer = BaseInferencer
    ModelType = ModelType


__all__ = ['BaseInferencer', 'ModelType']
