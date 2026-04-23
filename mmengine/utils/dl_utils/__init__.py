from .parrots_wrapper import _BatchNorm, _InstanceNorm
import platform
import sys
import torch
import torchvision


def collect_env():
    env_info = {
        'sys.platform': sys.platform,
        'Python': sys.version,
        'PyTorch': torch.__version__,
        'TorchVision': torchvision.__version__,
        'CUDA available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env_info['CUDA_HOME'] = torch.version.cuda
        env_info['GPU 0'] = torch.cuda.get_device_name(0)
    return env_info


def mmcv_full_available():
    return False


__all__ = ['_BatchNorm', '_InstanceNorm', 'collect_env', 'mmcv_full_available']
