import torch
import torch.jit as jit

from torch import Tensor
from torch.nn import Module


@jit.script
def widened_sigmoid(x: Tensor) -> Tensor:
    """Functional widened sigmoid activation
    
    Arguments:
        x (Tensor): input tensor

    Returns:
        x (Tensor): activated output tensor
    """
    EPS = 1e-3
    SCALE = 1. + 2. * EPS
    return .5 * (1. + SCALE * torch.tanh(.5 * x))


@jit.script
def shifted_softplus(x: Tensor) -> Tensor:
    """Functional shifted softplus activation
    
    Arguments:
        x (Tensor): input tensor

    Returns:
        x (Tensor): activated output tensor
    """
    sx = x - 1
    abs_sx = torch.abs(sx)
    return torch.log1p(torch.exp(-abs_sx)) + sx * (sx >= 0)


class WidenedSigmoid(Module):
    """Module widened sigmoid activation"""
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward widened sigmoid activation
    
        Arguments:
            x (Tensor): input tensor

        Returns:
            x (Tensor): activated output tensor
        """
        return widened_sigmoid(x)


class ShiftedSoftplus(Module):
    """Module shifted softplus activation"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward shifted softplus activation
    
        Arguments:
            x (Tensor): input tensor

        Returns:
            x (Tensor): activated output tensor
        """
        return shifted_softplus(x)
