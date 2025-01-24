from torch import nn
import chop.nn.snn.base as base
import torch


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


# class LinearUnfoldBias(Linear):
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         bias: bool = True,
#         step_mode="s",
#         config=None,
#     ) -> None:
#         """
#         Unfold bias across time steps (levels) for spiking linear layer.
#         This layer only supports multi-step mode, because the bias is shared across levels.

#         :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
#         :type step_mode: str

#         :param config: the configuration of the layer
#         :type config: dict
#         """
#         super().__init__(in_features, out_features, bias)
#         self.step_mode = step_mode
#         self.config = config
#         self.level = config["level"]

#     def forward(self, input):
#         output = super().forward(input)
#         if self.bias is not None:
#             output -= self.bias.unsqueeze(0)
#             output += self.bias.unsqueeze(0) / self.level

#         return output


# class LinearUnfoldBias(nn.Module):
#     def __init__(self, linear, **kwargs):
#         super(LinearUnfoldBias, self).__init__()
#         self.linear = linear
#         self.is_work = False
#         self.first = True
#         self.zero_output = None
#         self.neuron_type = kwargs["neuron_type"]
#         self.level = kwargs["level"]
#         self.steps = self.level
#         self.realize_time = self.steps

#     def reset(self):
#         # print("LLLinear reset")
#         self.is_work = False
#         self.first = True
#         self.zero_output = None
#         self.realize_time = self.steps

#     def forward(self, input):
#         # print("LLLinear.steps",self.steps)
#         x = input
#         # if x.ndim == 2:
#         #     B,N = x.shape
#         # elif x.ndim == 3:
#         #     B,C,N = x.shape
#         # N = self.linear.out_features
#         if x.dim() == 3:
#             B, N, _ = x.shape
#             D = self.linear.out_features
#             shape_new = (B, N, D)
#         elif x.dim() == 2:
#             B, _ = x.shape
#             D = self.linear.out_features
#             shape_new = (B, D)
#         if self.zero_output is None:
#             self.zero_output = torch.zeros(
#                 size=shape_new, device=x.device, dtype=x.dtype
#             )

#         if (not torch.is_tensor(x) and (x == 0.0)) or ((x == 0.0).all()):
#             self.is_work = False
#             if self.realize_time > 0:
#                 output = self.zero_output + (
#                     self.linear.bias.data.unsqueeze(0) / self.steps
#                     if self.linear.bias is not None
#                     else 0.0
#                 )
#                 self.realize_time = self.realize_time - 1
#                 self.is_work = True
#                 return output
#             return self.zero_output

#         output = self.linear(x)

#         if self.neuron_type == "IF":
#             pass
#         else:
#             if self.linear.bias is None:
#                 pass
#             else:
#                 output = output - self.linear.bias.data.unsqueeze(0)
#                 if self.realize_time > 0:
#                     output = output + self.linear.bias.data.unsqueeze(0) / self.steps
#                     self.realize_time = self.realize_time - 1

#         self.is_work = True
#         self.first = False

#         return output


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
        # N = self.linear.out_features
        if x.dim() == 3:
            B, N, _ = x.shape
            D = self.linear.out_features
            shape_new = (B, N, D)
        elif x.dim() == 2:
            B, _ = x.shape
            D = self.linear.out_features
            shape_new = (B, D)
        if self.zero_output is None:
            self.zero_output = torch.zeros(
                size=shape_new, device=x.device, dtype=x.dtype
            )

        if (not torch.is_tensor(x) and (x == 0.0)) or ((x == 0.0).all()):
            self.is_work = False
            if self.realize_time > 0:
                output = self.zero_output + (
                    self.linear.bias.data.unsqueeze(0) / self.steps
                    if self.linear.bias is not None
                    else 0.0
                )
                self.realize_time = self.realize_time - 1
                self.is_work = True
                return output
            return self.zero_output

        output = self.linear(x)

        if self.neuron_type == "IF":
            pass
        else:
            if self.linear.bias is None:
                pass
            else:
                output = output - self.linear.bias.data.unsqueeze(0)
                if self.realize_time > 0:
                    output = output + self.linear.bias.data.unsqueeze(0) / self.steps
                    self.realize_time = self.realize_time - 1

        self.is_work = True
        self.first = False

        return output
