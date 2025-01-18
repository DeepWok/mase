import torch.nn as nn
import torch
import numpy as np

class VoltageHook(nn.Module):
    def __init__(self, scale=1.0, momentum=0.1, mode='Max'):
        """
        * :ref:`API in English <VoltageHook.__init__-en>`
        .. _voltageHook.__init__-en:

        :param scale: initial scaling value
        :type scale: float
        :param momentum: momentum value
        :type momentum: float
        :param mode: The mode. Value "Max" means recording the maximum value of ANN activation, "99.9%" means recording the 99.9% precentile of ANN activation, and a float of 0-1 means recording the corresponding multiple of the maximum activation value.
        :type mode: str, float

        ``VoltageHook`` is placed behind ReLU and used to determine the range of activations in ANN inference.

        """
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale))
        self.mode = mode
        self.num_batches_tracked = 0
        self.momentum = momentum

    def forward(self, x):
        """
        * :ref:`API in English <VoltageHook.forward-en>`
        .. _VoltageHook.forward-en:

        :param x: input tensor
        :type x: torch.Tensor
        :return: original input tensor
        :rtype: torch.Tensor

        It doesn't process input tensors, but hooks the activation values of ReLU.

        """
        err_msg = 'You have used a non-defined VoltageScale Method.'
        if isinstance(self.mode, str):
            if self.mode[-1] == '%':
                try:
                    s_t = torch.tensor(np.percentile(x.detach().cpu(), float(self.mode[:-1])))
                except ValueError:
                    raise NotImplementedError(err_msg)
            elif self.mode.lower() in ['max']:
                s_t = x.max().detach()
            else:
                raise NotImplementedError(err_msg)
        elif isinstance(self.mode, float) and self.mode <= 1 and self.mode > 0:
            s_t = x.max().detach() * self.mode
        else:
            raise NotImplementedError(err_msg)
        
        if self.num_batches_tracked == 0:
            self.scale = s_t
        else:
            self.scale = (1 - self.momentum) * self.scale + self.momentum * s_t
        self.num_batches_tracked += x.shape[0]
        return x

class VoltageScaler(nn.Module):
    def __init__(self, scale=1.0):
        """
        * :ref:`API in English <VoltageScaler.__init__-en>`
        .. _VoltageScaler.__init__-en:

        :param scale: scaling value
        :type scale: float

        ``VoltageScaler`` is used for scaling current in SNN inference.

        """
        super().__init__()
        self.register_buffer('scale', torch.tensor(scale))

    def forward(self, x):
        """
        * :ref:`API in English <VoltageScaler.forward-en>`
        .. _VoltageScaler.forward-en:

        :param x: input tensor, or input current
        :type x: torch.Tensor
        :return: current after scaling
        :rtype: torch.Tensor

        """
        return x * self.scale

    def extra_repr(self):
        return '%f' % self.scale.item()