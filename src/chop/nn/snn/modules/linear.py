from torch import nn
import chop.nn.snn.base as base
import torch

from chop.nn.quantizers.SNN.LSQ import AlphaInit, ElasticBiSpiking


class Linear(nn.Linear, base.StepModule):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, step_mode="s"
    ) -> None:
        """
        * :ref:`API in English <Linear-en>`
        .. _Linear-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.Linear` for other parameters' API
        """
        super().__init__(in_features, out_features, bias)
        self.step_mode = step_mode


# TODO: Merge this with StepModule?
class LinearUnfoldBias(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        level: int = None,
        neuron_type: str = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            device,
            dtype,
        )
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.neuron_type = neuron_type
        self.level = level
        self.steps = self.level
        self.realize_time = self.steps

    def reset(self):
        # print("LLLinear reset")
        self.is_work = False
        self.first = True
        self.zero_output = None
        self.realize_time = self.steps

    def forward(self, input):
        # print("LLLinear.steps",self.steps)
        x = input
        # if x.ndim == 2:
        #     B,N = x.shape
        # elif x.ndim == 3:
        #     B,C,N = x.shape
        # N = self.out_features
        if x.dim() == 3:
            B, N, _ = x.shape
            D = self.out_features
            shape_new = (B, N, D)
        elif x.dim() == 2:
            B, _ = x.shape
            D = self.out_features
            shape_new = (B, D)
        if self.zero_output is None:
            self.zero_output = torch.zeros(
                size=shape_new, device=x.device, dtype=x.dtype
            )

        if (not torch.is_tensor(x) and (x == 0.0)) or ((x == 0.0).all()):
            self.is_work = False
            if self.realize_time > 0:
                output = self.zero_output + (
                    self.bias.data.unsqueeze(0) / self.steps
                    if self.bias is not None
                    else 0.0
                )
                self.realize_time = self.realize_time - 1
                self.is_work = True
                return output
            return self.zero_output

        output = super().forward(x)

        if self.neuron_type == "IF":
            pass
        else:
            if self.bias is None:
                pass
            else:
                output = output - self.bias.data.unsqueeze(0)
                if self.realize_time > 0:
                    output = output + self.bias.data.unsqueeze(0) / self.steps
                    self.realize_time = self.realize_time - 1

        self.is_work = True
        self.first = False

        return output


class LinearElasticBiSpiking(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        symmetric=True,
        config=None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias,
            device,
            dtype,
        )
        # NOTE: dead code from the original implementation (maybe useful in future reference)
        # self.weight_bits = config["weight_bits"]
        # self.quantize_act = config["quantize_act"]
        # self.register_buffer('weight_clip_val', torch.tensor([config["clip_val"]]))
        # self.input_bits = config["input_bits"]

        self.T = config["T"]
        self.act_clip_val = nn.ParameterList(
            [AlphaInit(torch.tensor(1.0), requires_grad=False) for i in range(self.T)]
        )
        self.act_quantizer = ElasticBiSpiking

    def forward(self, input):
        # quantize weight
        assert len(self.weight.size()) == 2

        weight = self.weight
        mem = torch.zeros_like(input[0]).cuda()
        output = torch.zeros_like(input).cuda()
        mem_old = 0
        for i in range(self.T):
            if i == 0:
                mem = input[0]
            else:
                # v = beta * mem_old (alpha - spike) + v_reset(which is 0) + input
                mem = (
                    mem_old
                    * 0.25
                    * (self.act_clip_val[i - 1].detach() - output[i - 1].detach())
                    + input[i]
                )

            # spike
            output[i] = self.act_quantizer.apply(
                mem, self.act_clip_val[i], self.input_bits, True
            )
            mem_old = mem.clone()

        out = nn.functional.linear(output, weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out
