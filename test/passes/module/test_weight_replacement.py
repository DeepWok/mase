"""weight_replacement copies parameters into the target without touching the source."""

import torch

from chop.passes.module.module_modify_helper import weight_replacement


def test_target_receives_an_independent_copy():
    source = torch.nn.Linear(4, 4)
    target = torch.nn.Linear(4, 4)
    source_weight = source.weight.detach().clone()
    source_bias = source.bias.detach().clone()

    weight_replacement(source, target)

    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)
    assert target.weight.data_ptr() != source.weight.data_ptr()

    # Mutating the target must not leak back into the source.
    target.weight.data.fill_(0.0)
    assert torch.equal(source.weight, source_weight)
    assert torch.equal(source.bias, source_bias)
