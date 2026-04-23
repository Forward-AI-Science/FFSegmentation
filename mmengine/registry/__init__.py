import importlib
import inspect
import sys
from typing import Any, Callable, Dict, List, Optional, Type, Union


class Registry:
    """A lightweight registry that mimics mmengine.registry.Registry."""

    def __init__(self, name: str, parent: Optional['Registry'] = None,
                 locations: Optional[List[str]] = None):
        self._name = name
        self._parent = parent
        self._locations = locations or []
        self._module_dict: Dict[str, type] = {}
        self._children: Dict[str, 'Registry'] = {}
        if parent is not None:
            parent._children[name] = self

    @property
    def name(self) -> str:
        return self._name

    def _do_register(self, name: str, module: type, force: bool = False):
        if not force and name in self._module_dict:
            raise KeyError(f"'{name}' is already registered in '{self._name}'")
        self._module_dict[name] = module

    def register_module(self, name=None, force: bool = False, module=None):
        if module is not None:
            n = name if isinstance(name, str) else module.__name__
            self._do_register(n, module, force)
            return module

        if callable(name) and not isinstance(name, type):
            # Used as @registry.register_module without ()
            cls = name
            self._do_register(cls.__name__, cls, force)
            return cls

        def _decorator(cls):
            n = name if isinstance(name, str) else cls.__name__
            self._do_register(n, cls, force)
            return cls

        return _decorator

    def _import_locations(self):
        for loc in self._locations:
            try:
                importlib.import_module(loc)
            except ImportError:
                pass

    def get(self, key: str) -> Optional[type]:
        if key in self._module_dict:
            return self._module_dict[key]
        # Try lazy-loading from locations
        self._import_locations()
        if key in self._module_dict:
            return self._module_dict[key]
        # Search parent
        if self._parent is not None:
            result = self._parent.get(key)
            if result is not None:
                return result
        # Search children (allows parent-level build to find child-registered classes)
        return self._search_children(key)

    def _search_children(self, key: str) -> Optional[type]:
        for child in self._children.values():
            child._import_locations()
            if key in child._module_dict:
                return child._module_dict[key]
            result = child._search_children(key)
            if result is not None:
                return result
        return None

    def build(self, cfg: Union[dict, Any], **default_kwargs) -> Any:
        if cfg is None:
            return None
        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a dict, got {type(cfg)}')
        from mmengine.config import ConfigDict
        cfg = cfg.copy()
        obj_type = cfg.pop('type')
        if isinstance(obj_type, str):
            cls = self.get(obj_type)
            if cls is None:
                raise KeyError(f"'{obj_type}' is not found in '{self._name}' registry")
        else:
            cls = obj_type
        # Convert nested plain dicts to ConfigDict for attribute-style access
        kwargs = {}
        for k, v in {**default_kwargs, **cfg}.items():
            if isinstance(v, dict) and not isinstance(v, ConfigDict):
                kwargs[k] = ConfigDict(v)
            else:
                kwargs[k] = v
        return cls(**kwargs)

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict or (
            self._parent is not None and key in self._parent)

    def __repr__(self) -> str:
        return (f'Registry(name={self._name}, '
                f'items={list(self._module_dict.keys())})')

    # Allow iteration over registered keys
    def __iter__(self):
        return iter(self._module_dict)


# ---- Singleton global registries ----

DATA_SAMPLERS = Registry('data_sampler')
DATASETS = Registry('dataset')
EVALUATOR = Registry('evaluator')
HOOKS = Registry('hook')
INFERENCERS = Registry('inferencer')
LOG_PROCESSORS = Registry('log_processor')
LOOPS = Registry('loop')
METRICS = Registry('metric')
MODEL_WRAPPERS = Registry('model_wrapper')
MODELS = Registry('model')
OPTIM_WRAPPER_CONSTRUCTORS = Registry('optim_wrapper_constructor')
OPTIM_WRAPPERS = Registry('optim_wrapper')
OPTIMIZERS = Registry('optimizer')
PARAM_SCHEDULERS = Registry('param_scheduler')
RUNNER_CONSTRUCTORS = Registry('runner_constructor')
RUNNERS = Registry('runner')
TASK_UTILS = Registry('task_util')
TRANSFORMS = Registry('transform')
VISBACKENDS = Registry('vis_backend')
VISUALIZERS = Registry('visualizer')
WEIGHT_INITIALIZERS = Registry('weight_initializer')


_default_scope: Optional[str] = None
_scope_registry: Dict[str, Registry] = {}


class DefaultScope:
    _current: Optional[str] = None

    def __init__(self, name: str, scope_name: str):
        self._name = name
        self._scope = scope_name

    @classmethod
    def get_current_instance(cls) -> Optional['DefaultScope']:
        return None

    @classmethod
    def get_instance(cls, name: str, scope_name: str) -> 'DefaultScope':
        inst = cls.__new__(cls)
        inst._name = name
        inst._scope = scope_name
        DefaultScope._current = scope_name
        return inst

    @property
    def scope_name(self) -> str:
        return self._scope


def init_default_scope(scope: str):
    global _default_scope
    _default_scope = scope


__all__ = [
    'Registry', 'DefaultScope', 'init_default_scope',
    'DATA_SAMPLERS', 'DATASETS', 'EVALUATOR', 'HOOKS', 'INFERENCERS',
    'LOG_PROCESSORS', 'LOOPS', 'METRICS', 'MODEL_WRAPPERS', 'MODELS',
    'OPTIM_WRAPPER_CONSTRUCTORS', 'OPTIM_WRAPPERS', 'OPTIMIZERS',
    'PARAM_SCHEDULERS', 'RUNNER_CONSTRUCTORS', 'RUNNERS', 'TASK_UTILS',
    'TRANSFORMS', 'VISBACKENDS', 'VISUALIZERS', 'WEIGHT_INITIALIZERS',
]
