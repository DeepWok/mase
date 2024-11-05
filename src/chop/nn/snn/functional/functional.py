import logging
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable, Union
from torch import Tensor
from chop.nn.snn.base import base


def multi_step_forward(
    x_seq: Tensor,
    single_step_module: Union[
        nn.Module, list[nn.Module], tuple[nn.Module], nn.Sequential, Callable
    ],
):
    """
    * :ref:`API in English <multi_step_forward-en>`

    .. _multi_step_forward-en:

    :param x_seq: the input tensor with ``shape=[T, batch_size, ...]``
    :type x_seq: torch.Tensor
    :param single_step_module: one or many single-step modules
    :type single_step_module: Union[nn.Module, list[nn.Module], tuple[nn.Module], nn.Sequential, Callable]
    :return: the output tensor with ``shape=[T, batch_size, ...]``
    :rtype: torch.torch.Tensor

    Applies multi-step forward on ``single_step_module``.

    """
    y_seq = []
    if isinstance(single_step_module, (list, tuple, nn.Sequential)):
        for t in range(x_seq.shape[0]):
            x_seq_t = x_seq[t]
            for m in single_step_module:
                x_seq_t = m(x_seq_t)
            y_seq.append(x_seq_t)
    else:
        for t in range(x_seq.shape[0]):
            y_seq.append(single_step_module(x_seq[t]))

    return torch.stack(y_seq)


def seq_to_ann_forward(
    x_seq: Tensor,
    stateless_module: Union[nn.Module, list, tuple, nn.Sequential, Callable],
):
    """
    * :ref:`API in English <seq_to_ann_forward-en>`

    .. _seq_to_ann_forward-en:

    :param x_seq: the input tensor with ``shape=[T, batch_size, ...]``
    :type x_seq: Tensor
    :param stateless_module: one or many stateless modules
    :type stateless_module: Union[nn.Module, list, tuple, nn.Sequential, Callable]
    :return: the output tensor with ``shape=[T, batch_size, ...]``
    :rtype: Tensor

    Applied forward on stateless modules.

    """
    y_shape = [x_seq.shape[0], x_seq.shape[1]]
    y = x_seq.flatten(0, 1)
    if isinstance(stateless_module, (list, tuple, nn.Sequential)):
        for m in stateless_module:
            y = m(y)
    else:
        y = stateless_module(y)
    y_shape.extend(y.shape[1:])
    return y.view(y_shape)