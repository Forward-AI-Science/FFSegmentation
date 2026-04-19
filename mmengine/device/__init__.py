import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


__all__ = ['get_device']
