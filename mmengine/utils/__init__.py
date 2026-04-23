import abc
import os
import subprocess
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

from .dl_utils import _BatchNorm, _InstanceNorm, collect_env, mmcv_full_available
from .dl_utils.parrots_wrapper import _BatchNorm, _InstanceNorm


def is_str(x: Any) -> bool:
    return isinstance(x, str)


def is_list_of(seq: Any, expected_type: Type) -> bool:
    if not isinstance(seq, list):
        return False
    return all(isinstance(item, expected_type) for item in seq)


def is_tuple_of(seq: Any, expected_type: Type) -> bool:
    if not isinstance(seq, tuple):
        return False
    return all(isinstance(item, expected_type) for item in seq)


def is_seq_of(seq: Any, expected_type: Type, seq_type=None) -> bool:
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    return all(isinstance(item, expected_type) for item in seq)


def to_2tuple(x):
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return tuple(x)
    return (x, x)


def mkdir_or_exist(dir_name: str, mode: int = 0o777) -> None:
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


def scandir(dir_path: str, suffix: Union[str, Tuple[str, ...], None] = None,
            recursive: bool = False) -> Iterable[str]:
    """Scan a directory and yield relative file paths.

    Args:
        dir_path: Root directory.
        suffix: Filter by extension(s), e.g. '.png' or ('.jpg', '.png').
        recursive: Whether to scan subdirectories.

    Yields:
        Relative file path strings from dir_path.
    """
    if suffix is not None and not isinstance(suffix, (str, tuple)):
        raise TypeError('suffix must be a string or tuple of strings')
    root = dir_path
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fname in sorted(filenames):
                rel = os.path.relpath(os.path.join(dirpath, fname), root)
                if suffix is None or rel.endswith(suffix):
                    yield rel
    else:
        for fname in sorted(os.listdir(root)):
            if os.path.isfile(os.path.join(root, fname)):
                if suffix is None or fname.endswith(suffix):
                    yield fname


def get_git_hash(fallback: str = 'unknown', digits: Optional[int] = None) -> str:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            h = result.stdout.strip()
            if digits is not None:
                h = h[:digits]
            return h
    except Exception:
        pass
    return fallback


__all__ = [
    'is_str', 'is_list_of', 'is_tuple_of', 'is_seq_of', 'to_2tuple',
    'mkdir_or_exist', 'scandir', 'get_git_hash', '_BatchNorm', '_InstanceNorm',
    'collect_env', 'mmcv_full_available',
]
