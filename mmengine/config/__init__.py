import argparse
import copy
import json
import os
import os.path as osp
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Union


class ConfigDict(dict):
    """A dict that supports attribute-style access and nested ConfigDict."""

    def __missing__(self, key):
        raise KeyError(key)

    def __getattr__(self, name: str):
        try:
            value = super().__getitem__(name)
        except KeyError:
            raise AttributeError(f"'ConfigDict' has no attribute '{name}'")
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
            super().__setitem__(name, value)
        return value

    def __setattr__(self, name: str, value: Any):
        super().__setitem__(name, value)

    def __delattr__(self, name: str):
        try:
            super().__delitem__(name)
        except KeyError:
            raise AttributeError(f"'ConfigDict' has no attribute '{name}'")

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.items():
            result[copy.deepcopy(k, memo)] = copy.deepcopy(v, memo)
        return result

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]


class Config:
    """Lightweight mmengine-compatible Config backed by ConfigDict."""

    @staticmethod
    def fromfile(filename: str) -> 'Config':
        filename = osp.abspath(osp.expanduser(str(filename)))
        if not osp.isfile(filename):
            raise FileNotFoundError(f'Config file not found: {filename}')
        ext = osp.splitext(filename)[1].lower()
        if ext not in ('.py', '.json', '.yaml', '.yml'):
            raise IOError(f'Unsupported config format: {ext}')
        cfg_dict = Config._parse(filename)
        return Config(cfg_dict, filename=filename)

    @staticmethod
    def _parse(filename: str) -> dict:
        ext = osp.splitext(filename)[1].lower()
        if ext == '.json':
            with open(filename, encoding='utf-8') as f:
                return json.load(f)
        elif ext in ('.yaml', '.yml'):
            import yaml
            with open(filename, encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        else:
            return Config._parse_py(filename)

    @staticmethod
    def _parse_py(filename: str) -> dict:
        cfg_dir = osp.dirname(osp.abspath(filename))
        namespace: dict = {'__name__': '__config__', '__file__': filename}
        # Add cfg_dir to path so _base_ relative imports work
        sys.path.insert(0, cfg_dir)
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                code = f.read()
            exec(compile(code, filename, 'exec'), namespace)
        except Exception as e:
            raise RuntimeError(f'Error parsing config {filename}: {e}') from e
        finally:
            if cfg_dir in sys.path:
                sys.path.remove(cfg_dir)

        cfg_dict = {k: v for k, v in namespace.items() if not k.startswith('__')}

        # Handle _base_ inheritance
        base_files = cfg_dict.pop('_base_', None)
        if base_files is not None:
            if isinstance(base_files, str):
                base_files = [base_files]
            base = {}
            for bf in base_files:
                bp = osp.join(cfg_dir, bf)
                bd = Config._parse(bp)
                Config._merge(bd, base)
            Config._merge(cfg_dict, base)
            cfg_dict = base

        return cfg_dict

    @staticmethod
    def _merge(src: dict, dst: dict) -> dict:
        for k, v in src.items():
            if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
                # Check for _delete_ flag
                if v.pop('_delete_', False):
                    dst[k] = v
                else:
                    Config._merge(v, dst[k])
            else:
                dst[k] = v
        return dst

    def __init__(self, cfg_dict: Optional[dict] = None, filename: Optional[str] = None):
        if cfg_dict is None:
            cfg_dict = {}
        object.__setattr__(self, '_cfg_dict', ConfigDict(cfg_dict))
        object.__setattr__(self, '_filename', filename)

    @property
    def filename(self):
        return object.__getattribute__(self, '_filename')

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, '_cfg_dict'), name)

    def __setattr__(self, name: str, value: Any):
        object.__getattribute__(self, '_cfg_dict')[name] = value

    def __delattr__(self, name: str):
        del object.__getattribute__(self, '_cfg_dict')[name]

    def __getitem__(self, key):
        return object.__getattribute__(self, '_cfg_dict')[key]

    def __setitem__(self, key, value):
        object.__getattribute__(self, '_cfg_dict')[key] = value

    def __delitem__(self, key):
        del object.__getattribute__(self, '_cfg_dict')[key]

    def __contains__(self, key):
        return key in object.__getattribute__(self, '_cfg_dict')

    def __iter__(self):
        return iter(object.__getattribute__(self, '_cfg_dict'))

    def __len__(self):
        return len(object.__getattribute__(self, '_cfg_dict'))

    def __repr__(self):
        return (f'Config(filename={self.filename}):\n'
                f'{dict(object.__getattribute__(self, "_cfg_dict"))}')

    def get(self, key, default=None):
        return object.__getattribute__(self, '_cfg_dict').get(key, default)

    def keys(self):
        return object.__getattribute__(self, '_cfg_dict').keys()

    def values(self):
        return object.__getattribute__(self, '_cfg_dict').values()

    def items(self):
        return object.__getattribute__(self, '_cfg_dict').items()

    def merge_from_dict(self, options: dict):
        if not isinstance(options, dict):
            raise TypeError(f'options must be dict, got {type(options)}')
        cfg_dict = object.__getattribute__(self, '_cfg_dict')
        for full_key, v in options.items():
            d = cfg_dict
            keys = full_key.split('.')
            for subkey in keys[:-1]:
                d = d.setdefault(subkey, ConfigDict())
            d[keys[-1]] = v

    def copy(self) -> 'Config':
        return Config(
            copy.deepcopy(object.__getattribute__(self, '_cfg_dict')),
            filename=object.__getattribute__(self, '_filename'))

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        return self.copy()

    @property
    def pretty_text(self) -> str:
        """Return a human-readable string of the config dict."""
        def _fmt(obj, indent=0):
            pad = '    ' * indent
            if isinstance(obj, dict):
                if not obj:
                    return '{}'
                lines = ['{']
                for k, v in obj.items():
                    lines.append(f'{pad}    {k!r}: {_fmt(v, indent+1)},')
                lines.append(pad + '}')
                return '\n'.join(lines)
            elif isinstance(obj, (list, tuple)):
                brace = ('(', ')') if isinstance(obj, tuple) else ('[', ']')
                if not obj:
                    return brace[0] + brace[1]
                lines = [brace[0]]
                for item in obj:
                    lines.append(f'{pad}    {_fmt(item, indent+1)},')
                lines.append(pad + brace[1])
                return '\n'.join(lines)
            else:
                return repr(obj)
        cfg_dict = dict(object.__getattribute__(self, '_cfg_dict'))
        return _fmt(cfg_dict)


class DictAction(argparse.Action):
    """argparse action: parse KEY=VALUE pairs into a dict."""

    @staticmethod
    def _parse_value(val: str):
        # Try int, float, bool, then string
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() == 'true':
            return True
        if val.lower() == 'false':
            return False
        if val.lower() == 'none':
            return None
        if (val.startswith('[') and val.endswith(']')) or \
           (val.startswith('(') and val.endswith(')')):
            is_tuple = val.startswith('(')
            inner = val[1:-1]
            items = [DictAction._parse_value(x.strip()) for x in inner.split(',') if x.strip()]
            return tuple(items) if is_tuple else items
        return val

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            k, _, v = kv.partition('=')
            options[k.strip()] = self._parse_value(v.strip())
        setattr(namespace, self.dest, options)


__all__ = ['Config', 'ConfigDict', 'DictAction']
