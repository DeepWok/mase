import math
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def floor_pass(x):
    y = x.floor()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LSQInteger(nn.Module):
    """
    LSQInteger is a PyTorch module for Learned Step Size Quantization (LSQ) with integer levels.
    Paper: https://arxiv.org/pdf/1902.08153
    Args:
        level (int): The number of quantization levels.
        sym (bool, optional): Whether to use symmetric quantization. Default is False.
        **kwargs: Additional keyword arguments.
    Attributes:
        s_init (float): Initial scale value.
        level (int): The number of quantization levels.
        sym (bool): Whether to use symmetric quantization.
        pos_max (torch.Tensor or str): The maximum positive value for quantization or "full" for full range.
        neg_min (torch.Tensor): The minimum negative value for quantization.
        s (torch.nn.Parameter): The learnable scale parameter.
        batch_init (int): Number of batches for initialization.
        init_state (int): Initialization state counter.
        debug (bool): Debug flag for TensorBoard logging.
        tfwriter (torch.utils.tensorboard.SummaryWriter or None): TensorBoard writer.
        global_step (float): Global step counter for TensorBoard logging.
        name (str): Name for TensorBoard logging.
    Methods:
        __repr__(): Returns a string representation of the object.
        reset(): Resets the internal state and history.
        forward(x): Forward pass for quantization.
    """

    def __init__(self, level, sym=False, **kwargs):
        super(LSQInteger, self).__init__()
        self.s_init = 0.0
        self.level = level
        self.sym = sym
        if level >= 256:
            self.pos_max = "full"
        else:
            if sym:
                self.pos_max = torch.tensor(float(level // 2 - 1))
                self.neg_min = torch.tensor(float(-level // 2))
            else:
                self.pos_max = torch.tensor(float(level - 1))
                self.neg_min = torch.tensor(float(0))

        self.s = nn.Parameter(torch.tensor(1.0))
        self.batch_init = 20
        self.init_state = 0
        self.debug = True
        self.tfwriter = kwargs["tfwriter"] if "tfwriter" in kwargs else None
        self.global_step = 0.0
        self.name = "LSQInteger"

    def __repr__(self):
        return f"LSQInteger(level={self.level}, sym={self.sym}, pos_max={self.pos_max}, neg_min={self.neg_min}, s={self.s.data})"

    def reset(self):
        self.history_max = torch.tensor(0.0)
        self.init_state = 0
        self.is_init = True

    def forward(self, x):
        if self.pos_max == "full":
            return x

        if str(self.neg_min.device) == "cpu":
            self.neg_min = self.neg_min.to(x.device)
        if str(self.pos_max.device) == "cpu":
            self.pos_max = self.pos_max.to(x.device)
        min_val = self.neg_min
        max_val = self.pos_max

        # according to LSQ, the grad scale should be proportional to sqrt(1/(quantize_state*neuron_number))
        s_grad_scale = 1.0 / ((max_val.detach().abs().mean() * x.numel()) ** 0.5)

        if self.init_state == 0 and self.training:
            self.s.data = torch.tensor(
                x.detach().abs().mean()
                * 2
                / (self.pos_max.detach().abs().mean() ** 0.5),
                dtype=torch.float32,
            ).cuda()
            self.init_state += 1
        elif self.init_state < self.batch_init and self.training:
            self.s.data = 0.9 * self.s.data + 0.1 * torch.tensor(
                torch.mean(torch.abs(x.detach()))
                * 2
                / (math.sqrt(max_val.detach().abs().mean())),
                dtype=torch.float32,
            )
            self.init_state += 1

        elif self.init_state == self.batch_init and self.training:
            self.init_state += 1
            print("initialize finish!!!!")

        s_scale = grad_scale(self.s, s_grad_scale)
        output = (
            torch.clamp(floor_pass(x / s_scale + 0.5), min=min_val, max=max_val)
            * s_scale
        )

        if self.debug and self.tfwriter is not None:
            self.tfwriter.add_histogram(
                tag="before_quan/".format(s_scale.item()) + self.name + "_data",
                values=(x).detach().cpu(),
                global_step=self.global_step,
            )
            self.tfwriter.add_histogram(
                tag="after_quan/".format(s_scale.item()) + self.name + "_data",
                values=(
                    (
                        torch.clamp(
                            floor_pass(x / s_scale + 0.5), min=min_val, max=max_val
                        )
                    )
                )
                .detach()
                .cpu(),
                global_step=self.global_step,
            )
            # self.debug = False
            self.tfwriter = None
            self.name = ""
            self.global_step = 0.0

        return output
