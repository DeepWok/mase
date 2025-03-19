from abc import abstractmethod
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging

from . import surrogate, base
from ...auto_cuda import neuron_kernel as ac_neuron_kernel
from ...auto_cuda import ss_neuron_kernel as ss_ac_neuron_kernel
try:
    import cupy
    from . import neuron_kernel, cuda_utils

except BaseException as e:
    logging.info(f'spikingjelly.activation_based.neuron: {e}')
    cupy = None
    neuron_kernel = None
    cuda_utils = None


class BaseNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False,
                 step_mode='s', backend='torch', store_v_seq: bool = False):
        """
        * :ref:`API in English <BaseNode.__init__-en>`

        .. _BaseNode.__init__-cn:

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，当神经元释放脉冲后，电压会被减去 ``v_threshold``
        :type v_reset: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        :param backend: 使用那种后端。不同的 ``step_mode`` 可能会带有不同的后端。可以通过打印 ``self.supported_backends`` 查看当前
            使用的步进模式支持的后端。在支持的情况下，使用 ``'cupy'`` 后端是速度最快的
        :type backend: str

        :param store_v_seq: 在使用 ``step_mode = 'm'`` 时，给与 ``shape = [T, N, *]`` 的输入后，是否保存中间过程的 ``shape = [T, N, *]``
            的各个时间步的电压值 ``self.v_seq`` 。设置为 ``False`` 时计算完成后只保留最后一个时刻的电压，即 ``shape = [N, *]`` 的 ``self.v`` 。
            通常设置成 ``False`` ，可以节省内存
        :type store_v_seq: bool

        可微分SNN神经元的基类神经元。

        * :ref:`中文API <BaseNode.__init__-cn>`

        .. _BaseNode.__init__-en:

        :param v_threshold: threshold of this neurons layer
        :type v_threshold: float

        :param v_reset: reset voltage of this neurons layer. If not ``None``, the neuron's voltage will be set to ``v_reset``
            after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
        :type v_reset: float

        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable

        :param detach_reset: whether detach the computation graph of reset in backward
        :type detach_reset: bool

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        :param backend: backend fot this neurons layer. Different ``step_mode`` may support for different backends. The user can
        print ``self.supported_backends`` and check what backends are supported by the current ``step_mode``. If supported,
        using ``'cupy'`` backend will have the fastest training speed
        :type backend: str

        :param store_v_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the voltage at each time-step to ``self.v_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the voltage at last time-step will be stored to ``self.v`` with ``shape = [N, *]``, which can reduce the
            memory consumption
        :type store_v_seq: bool

        This class is the base class of differentiable spiking neurons.
        """
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq

        # used in lava_exchange
        self.lava_s_cale = 1 << 6

        # used for cupy backend
        self.forward_kernel = None
        self.backward_kernel = None

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset
        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        """
         * :ref:`API in English <BaseNode.neuronal_charge-en>`

        .. _BaseNode.neuronal_charge-cn:

        定义神经元的充电差分方程。子类必须实现这个函数。

        * :ref:`中文API <BaseNode.neuronal_charge-cn>`

        .. _BaseNode.neuronal_charge-en:


        Define the charge difference equation. The sub-class must implement this function.
        """
        raise NotImplementedError

    def neuronal_fire(self):
        """
        * :ref:`API in English <BaseNode.neuronal_fire-en>`

        .. _BaseNode.neuronal_fire-cn:

        根据当前神经元的电压、阈值，计算输出脉冲。

        * :ref:`中文API <BaseNode.neuronal_fire-cn>`

        .. _BaseNode.neuronal_fire-en:


        Calculate out spikes of neurons by their current membrane potential and threshold voltage.
        """

        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        """
        * :ref:`API in English <BaseNode.neuronal_reset-en>`

        .. _BaseNode.neuronal_reset-cn:

        根据当前神经元释放的脉冲，对膜电位进行重置。

        * :ref:`中文API <BaseNode.neuronal_reset-cn>`

        .. _BaseNode.neuronal_reset-en:


        Reset the membrane potential according to neurons' output spikes.
        """
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)

        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def single_step_forward(self, x: torch.Tensor):
        """

        * :ref:`API in English <BaseNode.single_step_forward-en>`

        .. _BaseNode.single_step_forward-cn:

        :param x: 输入到神经元的电压增量
        :type x: torch.Tensor

        :return: 神经元的输出脉冲
        :rtype: torch.Tensor

        按照充电、放电、重置的顺序进行前向传播。

        * :ref:`中文API <BaseNode.single_step_forward-cn>`

        .. _BaseNode.single_step_forward-en:

        :param x: increment of voltage inputted to neurons
        :type x: torch.Tensor

        :return: out spikes of neurons
        :rtype: torch.Tensor

        Forward by the order of `neuronal_charge`, `neuronal_fire`, and `neuronal_reset`.

        """
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def single_step_forward_bool(self, x: torch.Tensor, input_bool=False, Fire_bool=False):
        self.v_float_to_tensor(x)
        if (input_bool):
            self.neuronal_charge(x)
        if (Fire_bool):
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
        else:
            spike = torch.zeros_like(x)

        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class One_LIFNode_convert(BaseNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False, timestep: int = 16, wait: int = 4, start_time: int = 0, biopolar_bool: bool = False):

        assert isinstance(tau, float)  # and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function,
                         detach_reset, step_mode, backend, store_v_seq)
        self.bioploar = biopolar_bool
        self.tau = tau
        self.decay_input = decay_input
        self.wait = wait
        self.timestep = timestep
        self.start_time = start_time

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch', 'cupy')
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}' + f', timestep={self.timestep}' + f', wait={self.wait},  starttime={self.start_time}'

    @staticmethod
    @torch.jit.script
    def one_spike_bool_decay(v: torch.Tensor,  tau: float):
        v = v * 2.0
        return v

    @staticmethod
    @torch.jit.script
    def one_spike_bool_input(x: torch.Tensor, v: torch.Tensor):

        v += x

        return v

    @staticmethod
    @torch.jit.script
    def one_spike_bool_fire(v: torch.Tensor, v_threshold: float, tau: float):
        v_threshold = v_threshold*2.0/tau
        spike = (v >= v_threshold).to(v)

        return spike, v, v_threshold

    @staticmethod
    @torch.jit.script
    def one_spike_bool_fire_bipolar(v: torch.Tensor, v_threshold: float, tau: float):
        v_threshold = v_threshold*2.0/tau
        spike = (v >= v_threshold).to(v)
        spike -= (v <= -v_threshold).to(v)

        return spike, v, v_threshold

    def single_step_forward(self, x: torch.Tensor,  time=0):

        if (x is not None):
            self.v_float_to_tensor(x)

        if (time >= self.start_time and time < self.start_time+self.timestep+self.wait):

            self.v = self.one_spike_bool_decay(self.v,  self.tau)

        if (time >= self.start_time and time < self.start_time + 1): ##change here 
            self.v = self.one_spike_bool_input(x, self.v)

        if (time >= self.wait+self.start_time and time < self.start_time+self.timestep+self.wait):
            if (self.bioploar):
                spike, self.v, self.v_threshold = self.one_spike_bool_fire_bipolar(
                    v=self.v,  v_threshold=self.v_threshold, tau=self.tau)
            else:
                spike, self.v, self.v_threshold = self.one_spike_bool_fire(
                    v=self.v,  v_threshold=self.v_threshold, tau=self.tau)

        else:
            spike = None

        return spike


class One_LIFNode(BaseNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False, timestep: int = 16, wait: int = 4, start_time: int = 0, biopolar_bool: bool = False):

        assert isinstance(tau, float)  # and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function,
                         detach_reset, step_mode, backend, store_v_seq)
        self.bipolar = biopolar_bool
        self.tau = tau
        self.decay_input = decay_input
        self.wait = wait
        self.timestep = timestep
        self.start_time = start_time

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch', 'cupy')
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}' + f', timestep={self.timestep}' + f', wait={self.wait},  starttime={self.start_time}'

    @staticmethod
    @torch.jit.script
    def one_spike_bool_decay(v: torch.Tensor,  tau: float):
        v = v * tau
        return v

    @staticmethod
    @torch.jit.script
    def one_spike_bool_input(x: torch.Tensor, v: torch.Tensor):

        v += x

        return v

    @staticmethod
    @torch.jit.script
    def one_spike_bool_fire(v: torch.Tensor, v_threshold: float):

        spike = (v >= v_threshold).to(v)

        return spike, v

    @staticmethod
    @torch.jit.script
    def one_spike_bool_fire_bipolar(v: torch.Tensor, v_threshold: float):

        spike = (v >= v_threshold).to(v)
        spike -= (v <= -v_threshold).to(v)

        return spike, v

    def single_step_forward(self, x: torch.Tensor,  time=0):

        if (x is not None):
            self.v_float_to_tensor(x)

        if (time >= self.start_time and time < self.start_time+self.timestep+self.wait):
            self.v = self.one_spike_bool_decay(self.v,  self.tau)

        if (time >= self.start_time and time < self.start_time + 1):

            self.v = self.one_spike_bool_input(x, self.v)

        if (time >= self.wait+self.start_time and time < self.start_time+self.timestep+self.wait):
            if (self.bipolar):
                spike, self.v = self.one_spike_bool_fire_bipolar(
                    v=self.v,  v_threshold=self.v_threshold)

            else:
                spike, self.v = self.one_spike_bool_fire(
                    v=self.v,  v_threshold=self.v_threshold)

        else:
            spike = None

        return spike


class neuron_trace(BaseNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False, timestep: int = 16, wait: int = 4, start_time: int = 0):

        assert isinstance(tau, float)  # and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function,
                         detach_reset, step_mode, backend, store_v_seq)

        self.tau = tau
        self.decay_input = decay_input
        self.wait = wait
        self.timestep = timestep
        self.start_time = start_time

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch', 'cupy')
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}' + f', timestep={self.timestep}' + f', wait={self.wait},  starttime={self.start_time}'

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input_reset0(x: torch.Tensor, v: torch.Tensor, tau: float):
        v = v * (1. / tau)*(1-x) + v * x

        return v

    def single_step_forward(self, x: torch.Tensor,  time=0):

        if (x is not None):
            if isinstance(self.v, float):
                self.v = 1/self.tau
            self.v_float_to_tensor(x)
        self.v = self.neuronal_charge_no_decay_input_reset0(
            x, self.v,  self.tau)
        return self.v


class WTA_neuron(BaseNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False, timestep: int = 16, wait: int = 4, start_time: int = 0, threshold_mv: float = 1.0):

        assert isinstance(tau, float)  # and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function,
                         detach_reset, step_mode, backend, store_v_seq)
        self.threshold_mv = threshold_mv
        self.tau = tau
        self.decay_input = decay_input
        self.wait = wait
        self.timestep = timestep
        self.start_time = start_time

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch', 'cupy')
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}' + f', timestep={self.timestep}' + f', wait={self.wait},  starttime={self.start_time}'

    @staticmethod
    @torch.jit.script
    def one_spike_bool_decay(v: torch.Tensor,  tau: float):
        v = v * tau
        return v

    @staticmethod
    @torch.jit.script
    def one_spike_bool_input(x: torch.Tensor, v: torch.Tensor):

        v += x

        return v

    @staticmethod
    @torch.jit.script
    def one_spike_bool_deacy_input(x: torch.Tensor, v: torch.Tensor, tau: float):

        v = v * tau + x

        return v

    @staticmethod
    @torch.jit.script
    def one_spike_bool_fire(v: torch.Tensor, v_threshold: float, tau: float, threshold_mv: float):
        spike = (v >= v_threshold).to(v)  # * (v//v_threshold)
        v_threshold = v_threshold-threshold_mv

        return spike, v, v_threshold

    def single_step_forward(self, x: torch.Tensor,  time=0):

        if (x is not None):
            self.v_float_to_tensor(x)

        if (time >= self.start_time and time < self.start_time+self.timestep):
            self.v = self.one_spike_bool_deacy_input(x, self.v, self.tau)

        if (time >= self.wait+self.start_time and time < self.start_time+self.timestep+self.wait):
            spike, self.v, self.v_threshold = self.one_spike_bool_fire(
                v=self.v,  v_threshold=self.v_threshold, tau=self.tau, threshold_mv=self.threshold_mv)

        else:
            spike = None

        return spike


class double_threshold_neuron(BaseNode):
    def __init__(self, tau: float = 2., decay_input: bool = False, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False, timestep: int = 16, wait: int = 4, start_time: int = 0, biopolar_bool: bool = False, threshold_mv: float = 1.0, threshold_shift: float = 1.0, scale=None):

        assert isinstance(tau, float)  # and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function,
                         detach_reset, step_mode, backend, store_v_seq)
        self.threshold_mv = threshold_mv
        self.threshold_shift = threshold_shift

        self.bipolar = biopolar_bool
        self.tau = tau
        self.decay_input = decay_input
        self.wait = wait
        self.timestep = timestep
        self.start_time = start_time
        self.scale = scale

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch', 'cupy')
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}' + f', timestep={self.timestep}' + f', wait={self.wait},  starttime={self.start_time}'

    @staticmethod
    @torch.jit.script
    def one_spike_bool_decay(v: torch.Tensor,  tau: float):
        v = v * tau
        return v

    @staticmethod
    @torch.jit.script
    def one_spike_bool_input(x: torch.Tensor, v: torch.Tensor):

        v += x

        return v

    @staticmethod
    @torch.jit.script
    def one_spike_bool_fire(v: torch.Tensor, v_threshold: float):

        spike = (v >= v_threshold).to(v)  # * (v//v_threshold)

        return spike, v

    @staticmethod
    @torch.jit.script
    def one_spike_bool_fire_bipolar(v: torch.Tensor, v_threshold: float):

        spike = (v >= v_threshold).to(v)  # * (v//v_threshold)
        spike -= (v <= -v_threshold).to(v)  # * (v//v_threshold)

        return spike, v

    @staticmethod
    @torch.jit.script
    def one_spike_wta_fire(v: torch.Tensor, threshold_shift: float, threshold_mv: float):
        spike = (v >= threshold_shift).to(v)  # * (v//v_threshold)
        threshold_shift = threshold_shift-threshold_mv

        return spike, threshold_shift

    def single_step_forward(self, x: torch.Tensor,  time=0):

        if (x is not None):
            self.v_float_to_tensor(x)

        if (time >= self.start_time and time < self.start_time+self.timestep+self.wait):
            self.v = self.one_spike_bool_decay(self.v,  self.tau)

        if (time >= self.start_time and time < self.start_time + self.timestep):

            self.v = self.one_spike_bool_input(x, self.v)

        if (time >= self.wait+self.start_time and time < self.start_time+self.timestep+self.wait):
            spike, self.v = self.one_spike_bool_fire(
                v=self.v,  v_threshold=self.v_threshold)
        elif (time >= self.wait+self.start_time+self.timestep and time < self.start_time+self.timestep+self.timestep+self.wait):
            spike, self.threshold_shift = self.one_spike_wta_fire(
                v=self.v*self.scale,  threshold_shift=self.threshold_shift, threshold_mv=self.threshold_mv)
        else:
            spike = None

        return spike
