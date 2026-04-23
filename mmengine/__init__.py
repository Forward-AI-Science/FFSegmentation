"""Pure-PyTorch replacement for mmengine.
Compatible with PyTorch 2.11+ and Python 3.12+.
"""

__version__ = '0.10.3'
version_info = (0, 10, 3)

from .config import Config, ConfigDict, DictAction
from .registry import DefaultScope, init_default_scope
from .utils import mkdir_or_exist, scandir

__all__ = [
    '__version__', 'version_info',
    'Config', 'ConfigDict', 'DictAction',
    'DefaultScope', 'init_default_scope',
    'mkdir_or_exist', 'scandir',
]
