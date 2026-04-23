import functools
import random
from typing import Any, Callable


def cache_randomness(func: Callable) -> Callable:
    """Decorator to cache random parameters for consistent augmentation."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        cache_attr = f'_cache_{func.__name__}'
        if not hasattr(self, '_use_cache') or not self._use_cache:
            return func(self, *args, **kwargs)
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, func(self, *args, **kwargs))
        return getattr(self, cache_attr)
    return wrapper


__all__ = ['cache_randomness']
