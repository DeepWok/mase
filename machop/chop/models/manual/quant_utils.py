from typing import Callable

from chop.passes.transforms.quantize.quant_parsers import parse_node_config
from chop.passes.transforms.quantize.quantized_funcs import quantized_func_map
from chop.passes.transforms.quantize.quantized_modules import quantized_module_map


def get_quantized_cls(mase_op: str, quant_config: dict) -> type:
    quant_arith = quant_config["name"]
    return quantized_module_map[f"{mase_op}_{quant_arith}"]


def get_quantized_func(mase_op: str, quant_config: dict) -> Callable:
    quant_arith = quant_config["name"]
    return quantized_func_map[f"{mase_op}_{quant_arith}"]


def parse_op_quant_config(mase_op: str, config: dict) -> dict:
    return parse_node_config(config=config, mase_op=mase_op)
