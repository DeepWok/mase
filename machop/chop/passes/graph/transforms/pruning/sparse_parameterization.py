import torch
import pdb

# Parametrizations
class FakeSparseWeight(torch.nn.Module):
    r"""Parametrization for the weights. Should be attached to the 'weight' or
    any other parameter that requires a mask applied to it.

    Note::

        Once the mask is passed, the variable should not change the id. The
        contents of the mask can change, but the mask reference itself should
        not.
    """

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x):
        #pdb.set_trace()
        try:
            self.mask = self.mask.expand(x.shape)
        except:
            if len(self.mask.shape) == 2:
                self.mask = self.mask.view(self.mask.shape[0],self.mask.shape[1], 1, 1).expand(-1,-1,3,3)
            if len(self.mask.shape) == 1:
                self.mask = self.mask.view(self.mask.shape[0], 1, 1, 1).expand(-1,x.shape[1],3,3)
        assert self.mask.shape == x.shape
        #pdb.set_trace()
        return self.mask * x

    def state_dict(self, *args, **kwargs):
        # This is a hack
        # We don't want to let the parametrizations to save the mask.
        # That way we make sure that the linear module doesn't store the masks
        # alongside their parametrizations.
        return {}


# Structured Pruning Parameterizations
class FakeStructuredSparseWeight(torch.nn.Module):
    r"""
    Parametrization for Structured Pruning. Like FakeSparsity, this should be attached to
    the  'weight' or any other parameter that requires a mask.

    Instead of an element-wise bool mask, this parameterization uses a row-wise bool mask.
    """

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x):
        assert isinstance(self.mask, torch.Tensor)
        assert self.mask.shape[0] == x.shape[0]
        shape = [1] * len(x.shape)
        shape[0] = -1
        return self.mask.reshape(shape) * x

    def state_dict(self, *args, **kwargs):
        # avoid double saving masks
        return {}
