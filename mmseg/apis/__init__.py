# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_model, init_model, show_result_pyplot

try:
    from .mmseg_inferencer import MMSegInferencer
    from .remote_sense_inferencer import RSImage, RSInferencer
    _extra = ['MMSegInferencer', 'RSInferencer', 'RSImage']
except ImportError:
    MMSegInferencer = None
    RSInferencer = None
    RSImage = None
    _extra = []

__all__ = ['init_model', 'inference_model', 'show_result_pyplot'] + _extra
