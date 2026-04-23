import io
import logging
import os
import os.path as osp
import re
import warnings
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


class CheckpointLoader:
    """Load checkpoints from various sources."""

    _schemes: dict = {}

    @classmethod
    def register_scheme(cls, prefixes, loader=None, force=False):
        def _register(fn):
            for prefix in ([prefixes] if isinstance(prefixes, str) else prefixes):
                cls._schemes[prefix] = fn
            return fn
        if loader is not None:
            return _register(loader)
        return _register

    @classmethod
    def load_checkpoint(cls, filename: str, map_location=None, logger=None):
        filename = filename.strip()
        for prefix, loader in cls._schemes.items():
            if filename.startswith(prefix):
                return loader(filename, map_location=map_location)
        # Default: local file
        return torch.load(filename, map_location=map_location, weights_only=False)


def _load_checkpoint(filename: str, map_location=None, logger=None):
    filename = str(filename)
    if filename.startswith(('http://', 'https://')):
        return _load_checkpoint_from_url(filename, map_location)
    return CheckpointLoader.load_checkpoint(filename, map_location, logger)


def _load_checkpoint_from_url(url: str, map_location=None):
    """Download checkpoint from URL to a temp file then load it."""
    import os, tempfile, urllib.request
    cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'mmseg', 'checkpoints')
    os.makedirs(cache_dir, exist_ok=True)
    filename = url.split('/')[-1]
    cached_path = os.path.join(cache_dir, filename)
    if not os.path.isfile(cached_path):
        print(f'Downloading checkpoint from {url} ...')
        urllib.request.urlretrieve(url, cached_path)
        print(f'Saved to {cached_path}')
    return torch.load(cached_path, map_location=map_location, weights_only=False)


def load_state_dict(module: nn.Module, state_dict: Dict[str, Any],
                    strict: bool = False, logger=None):
    unexpected_keys: list = []
    missing_keys: list = []
    own_state = module.state_dict()

    for name, param in state_dict.items():
        if name in own_state:
            try:
                own_state[name].copy_(param)
            except Exception:
                pass
        else:
            unexpected_keys.append(name)

    for name in own_state:
        if name not in state_dict:
            missing_keys.append(name)

    if missing_keys:
        warnings.warn(f'Missing keys: {missing_keys}')
    if unexpected_keys:
        warnings.warn(f'Unexpected keys: {unexpected_keys}')


def _load_checkpoint_to_model(model: nn.Module, checkpoint: dict,
                               strict: bool = False, logger=None,
                               revise_keys=None):
    state_dict = checkpoint.get('state_dict', checkpoint)
    if revise_keys:
        for p, r in revise_keys:
            state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}
    load_state_dict(model, state_dict, strict=strict, logger=logger)


def load_checkpoint(model: nn.Module, filename: str,
                    map_location: Optional[str] = 'cpu',
                    strict: bool = False,
                    logger=None,
                    prefix: Optional[str] = None,
                    revise_keys=None) -> dict:
    checkpoint = _load_checkpoint(filename, map_location=map_location,
                                  logger=logger)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f'Checkpoint must be a dict, got {type(checkpoint)}')

    state_dict = checkpoint.get('state_dict', checkpoint)

    # Strip prefix if requested
    if prefix:
        state_dict = {
            k[len(prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }

    if revise_keys:
        for p, r in revise_keys:
            state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}

    load_state_dict(model, state_dict, strict=strict, logger=logger)
    return checkpoint


__all__ = [
    'CheckpointLoader', '_load_checkpoint', 'load_state_dict',
    '_load_checkpoint_to_model', 'load_checkpoint',
]
