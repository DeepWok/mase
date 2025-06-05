
import torch
import torch.nn as nn
from a_cx_mxint_quant.quantizers import mxint_hardware
from chop.tools import get_logger
from chop.tools.logger import set_logging_verbosity

logger = get_logger(__name__)
set_logging_verbosity("debug")

def test_mxint_linear():
    """
    This is just a description of how it actually works for mxint.
    The actual implementation can be checked at a_cx_mxint_quant/linear.py file.
    """

    q_config = {
                "quant_type": "mxint",
                "data_in_width": 8,
                "data_in_exponent_width": 8,
                "data_in_parallelism": (16, 16),
                "data_out_width": 8,
                "data_out_exponent_width": 8,
                "data_out_parallelism": (16, 16),
                "weight_width": 6,
                "weight_exponent_width": 8,
                "weight_parallelism": (16, 16), # Note: weight_parallelism is only used in layer_norm
                "bias_width": 6,
                "bias_exponent_width": 8,
                "bias_parallelism": (1, 16), # Note: bias_parallelism is only used in layer_norm
    }
    x = torch.randn(1, 32, 1024)
    q_x, mant_x, exp_x = mxint_hardware(x, {
        "width": q_config["data_in_width"],
        "exponent_width": q_config["data_in_exponent_width"]
    }, parallelism = q_config["data_in_parallelism"])    

    logger.info(f'the mant_input: {mant_x}, the exp_input: {exp_x}')
    layer = nn.Linear(1024, 32)

    weight = layer.weight
    q_w, mant_q_w, exp_q_w = mxint_hardware(weight, {
        "width": q_config["weight_width"],
        "exponent_width": q_config["weight_exponent_width"]
    }, parallelism = q_config["weight_parallelism"])
    logger.info(f'the mant_weight: {mant_q_w}, the exp_weight: {exp_q_w}')

    bias = layer.bias
    q_b, mant_q_b, exp_q_b = mxint_hardware(bias, {
        "width": q_config["bias_width"],
        "exponent_width": q_config["bias_exponent_width"]
    }, parallelism = q_config["bias_parallelism"])
    logger.info(f'the mant_bias: {mant_q_b}, the exp_bias: {exp_q_b}')

    q_o = q_x @ q_w.T + q_b

    q_o, mant_q_o, exp_q_o = mxint_hardware(q_o, {
        "width": q_config["data_out_width"],
        "exponent_width": q_config["data_out_exponent_width"]
    }, parallelism = q_config["data_out_parallelism"])
    logger.info(f'the mant_output: {mant_q_o}, the exp_output: {exp_q_o}')

    from a_cx_mxint_quant.utils import _get_similarity
    similarity = _get_similarity(q_o, layer(x), metric="cosine")
    logger.info(f'the similarity: {similarity.mean()}')


if __name__ == "__main__":
    test_mxint_linear()

