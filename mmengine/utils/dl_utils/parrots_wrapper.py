import torch.nn as nn

_BatchNorm = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
_InstanceNorm = (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
