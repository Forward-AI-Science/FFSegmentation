import logging
import sys
from typing import Optional

_logger_dict: dict = {}
_current_instance: Optional['MMLogger'] = None


class MMLogger(logging.Logger):
    def __init__(self, name: str, log_file: Optional[str] = None,
                 log_level: int = logging.INFO, **kwargs):
        super().__init__(name, log_level)
        if not self.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.addHandler(handler)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.addHandler(fh)
        _logger_dict[name] = self
        global _current_instance
        _current_instance = self

    @classmethod
    def get_current_instance(cls) -> 'MMLogger':
        global _current_instance
        if _current_instance is None:
            _current_instance = cls('mmseg')
        return _current_instance

    @classmethod
    def get_instance(cls, name: str, **kwargs) -> 'MMLogger':
        if name in _logger_dict:
            return _logger_dict[name]
        return cls(name, **kwargs)


class MessageHub:
    _instances: dict = {}

    def __init__(self, name: str = 'default'):
        self.name = name
        self._log_scalars: dict = {}
        self._runtime_info: dict = {}
        MessageHub._instances[name] = self

    @classmethod
    def get_current_instance(cls) -> 'MessageHub':
        if 'default' not in cls._instances:
            cls._instances['default'] = cls()
        return cls._instances['default']

    @classmethod
    def get_instance(cls, name: str) -> 'MessageHub':
        if name not in cls._instances:
            cls._instances[name] = cls(name)
        return cls._instances[name]

    def update_scalars(self, log_dict: dict):
        self._log_scalars.update(log_dict)

    def update_info(self, key: str, value):
        self._runtime_info[key] = value

    def get_info(self, key: str, default=None):
        return self._runtime_info.get(key, default)

    @property
    def log_scalars(self):
        return self._log_scalars

    @property
    def runtime_info(self):
        return self._runtime_info


def print_log(msg, logger=None, level: int = logging.INFO):
    if logger is None or logger == 'current':
        log = MMLogger.get_current_instance()
    elif isinstance(logger, str):
        log = MMLogger.get_instance(logger)
    elif isinstance(logger, logging.Logger):
        log = logger
    else:
        print(msg)
        return
    log.log(level, msg)


__all__ = ['MMLogger', 'MessageHub', 'print_log']
