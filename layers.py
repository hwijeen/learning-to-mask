"""Contains novel layer definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""
    @staticmethod
    def forward(ctx, inputs: Tensor, threshold: float) -> Tensor:
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        return outputs

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output, None


class Ternarizer(torch.autograd.Function):
    """Ternarizes {-1, 0, 1} a real valued tensor."""
    @staticmethod
    def forward(ctx, inputs: Tensor, threshold: float) -> Tensor:
        outputs = inputs.clone()
        outputs.fill_(0)
        outputs[inputs < 0] = -1
        outputs[inputs > threshold] = 1
        return outputs

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output, None


class MaskedLinear(nn.Module):
    def __init__(self, weight: Tensor, bias: Tensor,
                 mask_init: str = 'uniform', mask_scale: float = 2e-2,
                 threshold_fn:str = 'binarizer', threshold: float = 1e-2,
                 initial_sparsity: float = 0.1, device=None, dtype=None) -> None:
        super().__init__()  # call grandparent's init
        self.weight = weight
        self.bias = bias
        self.mask_init = mask_init
        self.mask_scale = mask_scale
        self.threshold_fn = threshold_fn
        self.threshold = threshold
        self.initial_sparsity = initial_sparsity

        # Initialize real-valued mask weights.
        self.mask_real = self.weight.data.new(self.weight.size())
        if mask_init == '1s':
            self.mask_real.fill_(mask_scale)
        elif mask_init == 'uniform':
            # set right scale so that threhold equals initial_spasity
            left_scale = -1 * mask_scale
            right_scale = (mask_scale + threshold) / initial_sparsity - mask_scale
            # self.mask_real.uniform_(-1 * mask_scale, mask_scale)
            self.mask_real.uniform_(left_scale, right_scale)
        # mask_real is now a trainable parameter.
        self.mask_real = Parameter(self.mask_real)

        # Initialize the thresholder.
        if threshold_fn == 'binarizer':
            self.threshold_fn = Binarizer()
        elif threshold_fn == 'ternarizer':
            self.threshold_fn = Ternarizer()

    def forward(self, input: Tensor) -> Tensor:
        # Get binarized/ternarized mask from real-valued mask.
        mask_thresholded = self.threshold_fn.apply(self.mask_real, self.threshold)
        # Mask weights with above mask.
        weight_thresholded = mask_thresholded * self.weight
        # Get output using modified weight.
        return F.linear(input, weight_thresholded, self.bias)

    # .to uses this
    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        self.bias.data = fn(self.bias.data)

    @property
    def num_zeros(self):
        return self.mask_real.le(self.threshold).sum().item()
