import torch, sys

sys.path.append("/workspace/machop/")
from torch import Tensor
from chop.passes.transforms.quantize.quantizers.integer import _integer_quantize


def quantize_to_int(x: Tensor, width: int, frac_width: int):
    x = _integer_quantize(x, width, frac_width) * (2**frac_width)
    x = x.int() & (2**width - 1)
    return x


def twos_complement_to_float(binary_string: str, width: int, frac_width: int):
    # Determine the sign
    sign_bit = binary_string[0]

    # Extract integer and fractional parts
    integer_part = binary_string[1 : 1 + width]

    # Calculate integer magnitude
    integer_magnitude = int(integer_part, 2)

    # Apply two's complement conversion for negative numbers
    if sign_bit == "1":
        integer_magnitude = -(2 ** (width - 1)) + integer_magnitude

    # Calculate scaling factor
    scaling_factor = 2**frac_width

    # Calculate floating-point value
    float_value = integer_magnitude / scaling_factor

    return float_value


def generate_table_software(scale, width, frac_width, out_width, out_frac_width):
    addr = torch.tensor(range(0, 2 ** (width)), dtype=int)
    table = torch.zeros(2 ** (width))
    for i in range(2 ** (width)):
        element = twos_complement_to_float(
            format(addr[i], f"0{width}b"), width, frac_width
        )
        table[i] = element * scale
    table = _integer_quantize(table.exp(), out_width, out_frac_width)
    return table


def generate_table_hardware(scale, width, frac_width, out_width, out_frac_width):
    addr = torch.tensor(range(0, 2 ** (width)), dtype=int)
    table = torch.zeros(2 ** (width))
    for i in range(2 ** (width)):
        element = twos_complement_to_float(
            format(addr[i], f"0{width}b"), width, frac_width
        )
        table[i] = element * scale
    table = quantize_to_int(table.exp(), out_width, out_frac_width)
    return table


def generate_table_div_hardware(width, out_width, out_frac_width):
    addr = torch.tensor(range(0, 2 ** (width - 1)), dtype=int)
    table = torch.zeros(2 ** (width - 1))
    for i in range(2 ** (width - 1)):
        element = twos_complement_to_float(format(addr[i], f"0{width}b"), width, 0)
        table[i] = element
    table = quantize_to_int(1 / table, out_width, out_frac_width)
    return table


def generate_table_div_software(width, out_width, out_frac_width):
    addr = torch.tensor(range(0, 2 ** (width - 1)), dtype=int)
    table = torch.zeros(2 ** (width - 1))
    for i in range(2 ** (width - 1)):
        element = twos_complement_to_float(format(addr[i], f"0{width}b"), width, 0)
        table[i] = element
    table = _integer_quantize(1 / table, out_width, out_frac_width)
    return table


class QHashSoftmax(torch.nn.Module):
    def __init__(
        self,
        config,
    ):
        super(QHashSoftmax, self).__init__()
        self.in_width = config["data_in_width"]
        self.in_frac_width = config["data_in_frac_width"]
        self.exp_width = config["exp_width"]
        self.exp_frac_width = config["exp_frac_width"]
        self.out_width = config["data_out_width"]
        self.out_frac_width = config["data_out_frac_width"]
        self.div_width = config["div_width"]

    def forward(self, x, scale):
        table_exp = generate_table_software(
            scale,
            self.in_width,
            self.in_frac_width,
            self.exp_width,
            self.exp_frac_width,
        )

        table_div = generate_table_div_software(
            self.div_width + 1, self.out_width, self.out_frac_width
        )
        x = quantize_to_int(x, self.in_width, self.in_frac_width)
        exp = table_exp[x]
        exp_sum = exp.sum(dim=-1, keepdim=True)
        # quantize to div_width
        one_over_div = _integer_quantize(exp_sum // exp, self.div_width + 1, 0)
        one_over_div = torch.where(
            exp == 0, torch.tensor(2**self.div_width - 1), one_over_div
        )
        one_over_div = torch.tensor(one_over_div, dtype=int)

        div = table_div[one_over_div]
        return div
