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


# ========================================================================================================
# SNN quantization from SpikeZIP-TF
# ========================================================================================================
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


# ========================================================================================================
# SNN quantization from SpikeLM
# ========================================================================================================
class AlphaInit(nn.Parameter):
    def __init__(self, tensor, requires_grad=True):
        super(AlphaInit, self).__new__(
            nn.Parameter, data=tensor, requires_grad=requires_grad
        )
        self.initialized = False

    def _initialize(self, init_tensor):
        assert not self.initialized, "already initialized."
        self.data.copy_(init_tensor)
        self.initialized = True

    def initialize_wrapper(self, tensor, num_bits, symmetric, init_method="default"):
        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** (num_bits) - 1
        if Qp == 0:
            Qp = 1.0
        if init_method == "default":
            init_val = (
                2 * tensor.abs().mean() / math.sqrt(Qp)
                if symmetric
                else 4 * tensor.abs().mean() / math.sqrt(Qp)
            )
        elif init_method == "uniform":
            init_val = 1.0 / (2 * Qp + 1) if symmetric else 1.0 / Qp

        self._initialize(init_val)


class ElasticBiSpiking(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input
        elif num_bits == 1 or num_bits == 2:
            Qn = -1
            Qp = 1

        eps = torch.tensor(0.00001).float().to(alpha.device)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(
                input, num_bits, symmetric=True, init_method="default"
            )
        alpha = torch.where(alpha > eps, alpha, eps)
        assert alpha > 0, "alpha = {:.6f} becomes non-positive".format(alpha)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp
        if num_bits == 1:
            q_w = input.sign()  ################################## binary
        else:
            q_w = (input / alpha).round().clamp(Qn, Qp)  ###################### ternary
        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)
        if ctx.num_bits == 1:
            grad_alpha = (
                ((input_.sign()) * grad_output * grad_scale).sum().unsqueeze(dim=0)
            )
        else:
            grad_alpha = (
                (
                    (
                        indicate_small * Qn
                        + indicate_big * Qp
                        + indicate_middle * (-q_w + q_w.round())
                    )
                    * grad_output
                    * grad_scale
                )
                .sum()
                .unsqueeze(dim=0)
            )
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None
