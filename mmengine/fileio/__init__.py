import json
import os
import pickle
from pathlib import Path
from typing import Any, Optional, Union


def load(file: Union[str, Path], file_format: Optional[str] = None, **kwargs) -> Any:
    if isinstance(file, (str, Path)):
        file = str(file)
        if file_format is None:
            ext = file.rsplit('.', 1)[-1].lower() if '.' in file else 'pkl'
            file_format = ext
        if file_format == 'json':
            with open(file, encoding='utf-8') as f:
                return json.load(f)
        elif file_format in ('pkl', 'pickle'):
            with open(file, 'rb') as f:
                return pickle.load(f)
        elif file_format in ('yaml', 'yml'):
            import yaml
            with open(file, encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif file_format == 'txt':
            with open(file, encoding='utf-8') as f:
                return [ln.rstrip('\n') for ln in f]
        else:
            with open(file, 'rb') as f:
                return f.read()
    else:
        return pickle.load(file)


def save(obj: Any, file: Union[str, Path], file_format: Optional[str] = None, **kwargs):
    if isinstance(file, (str, Path)):
        file = str(file)
        if file_format is None:
            ext = file.rsplit('.', 1)[-1].lower() if '.' in file else 'pkl'
            file_format = ext
        parent = os.path.dirname(file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if file_format == 'json':
            with open(file, 'w', encoding='utf-8') as f:
                json.dump(obj, f, indent=2)
        elif file_format in ('pkl', 'pickle'):
            with open(file, 'wb') as f:
                pickle.dump(obj, f)
        elif file_format in ('yaml', 'yml'):
            import yaml
            with open(file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(obj, f)
    else:
        pickle.dump(obj, file)


def get(filepath: Union[str, Path], backend_args: Optional[dict] = None) -> bytes:
    with open(str(filepath), 'rb') as f:
        return f.read()


__all__ = ['load', 'save', 'get']
