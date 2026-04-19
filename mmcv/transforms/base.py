from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional


class BaseTransform(metaclass=ABCMeta):
    """Abstract base class for all mmcv-style transforms."""

    @abstractmethod
    def transform(self, results: Dict) -> Optional[Dict]:
        pass

    def __call__(self, results: Dict) -> Optional[Dict]:
        return self.transform(results)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


__all__ = ['BaseTransform']
