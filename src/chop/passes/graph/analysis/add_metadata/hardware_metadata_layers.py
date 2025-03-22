from typing import Dict, TypedDict, Literal


class IpDescType(TypedDict):
    name: str
    dependence_files: list[str]


supported_hw_quantisations = Literal["fixed", "mxint"]

# mase_op : the set of functional equivalent IPs with different design configurations.
# The first IP in each list is used by default
norm: IpDescType = {
    "name": "norm",
    "dependence_files": [
        "common/rtl/join2.sv",
        "common/rtl/split2.sv",
        "common/rtl/register_slice.sv",
        "common/rtl/lut.sv",
        "memory/rtl/simple_dual_port_ram.sv",
        "memory/rtl/repeat_circular_buffer.sv",
        "memory/rtl/fifo.sv",
        "memory/rtl/skid_buffer.sv",
        "cast/rtl/floor_round.sv",
        "cast/rtl/signed_clamp.sv",
        "cast/rtl/fixed_signed_cast.sv",
        "linear_layers/fixed_operators/rtl/fixed_accumulator.sv",
        "linear_layers/fixed_operators/rtl/fixed_adder_tree.sv",
        "linear_layers/fixed_operators/rtl/fixed_adder_tree_layer.sv",
        "linear_layers/fixed_operators/rtl/fixed_lut_index.sv",
        "linear_layers/fixed_operators/rtl/fixed_range_augmentation.sv",
        "linear_layers/fixed_operators/rtl/fixed_range_reduction.sv",
        "scalar_operators/fixed/rtl/fixed_isqrt.sv",
        "scalar_operators/fixed/rtl/fixed_nr_stage.sv",
        "linear_layers/matmul/rtl/matrix_fifo.sv",
        "linear_layers/matmul/rtl/matrix_flatten.sv",
        "linear_layers/matmul/rtl/matrix_unflatten.sv",
        "normalization_layers/rtl/channel_selection.sv",
        "normalization_layers/rtl/group_norm_2d.sv",
        "normalization_layers/rtl/rms_norm_2d.sv",
        "normalization_layers/rtl/batch_norm_2d.sv",
        "normalization_layers/rtl/norm.sv",
    ],
}


INTERNAL_COMP: Dict[str, Dict[supported_hw_quantisations, IpDescType]] = {
    "linear": {
        "fixed": {
            "name": "fixed_linear",
            "dependence_files": [
                "cast/rtl/fixed_cast.sv",
                "linear_layers/fixed_operators/rtl/fixed_dot_product.sv",
                "linear_layers/fixed_operators/rtl/fixed_vector_mult.sv",
                "linear_layers/fixed_operators/rtl/fixed_accumulator.sv",
                "linear_layers/fixed_operators/rtl/fixed_adder_tree.sv",
                "linear_layers/fixed_operators/rtl/fixed_adder_tree_layer.sv",
                "linear_layers/fixed_operators/rtl/fixed_mult.sv",
                "common/rtl/register_slice.sv",
                "common/rtl/join2.sv",
                "memory/rtl/unpacked_repeat_circular_buffer.sv",
                "memory/rtl/skid_buffer.sv",
                "linear_layers/fixed_linear_layer/rtl/fixed_linear.sv",
                "linear_layers/matmul/rtl/matrix_flatten.sv",
                "linear_layers/matmul/rtl/matrix_unflatten.sv",
                "linear_layers/matmul/rtl/matrix_fifo.sv",
                "linear_layers/matmul/rtl/matrix_accumulator.sv",
                "linear_layers/matmul/rtl/simple_matmul.sv",
                "linear_layers/matmul/rtl/matmul.sv",
                "linear_layers/matmul/rtl/transpose.sv",
                "linear_layers/matmul/rtl/matrix_stream_transpose.sv",
            ],
        },
        "mxint": {
            "name": "mxint_linear",
            "dependence_files": [
                "linear_layers/mxint_operators/rtl/mxint_linear.sv",
                "linear_layers/mxint_operators/rtl/mxint_circular.sv",
                "memory/rtl/input_buffer.sv",
                "linear_layers/mxint_operators/rtl/mxint_dot_product.sv",
                "linear_layers/mxint_operators/rtl/mxint_accumulator.sv",
                "linear_layers/mxint_operators/rtl/mxint_cast.sv",
                "linear_layers/mxint_operators/rtl/log2_max_abs.sv",
                "linear_layers/mxint_operators/rtl/or_tree.sv",
                "linear_layers/mxint_operators/rtl/or_tree_layer.sv",
                "linear_layers/mxint_operators/rtl/mxint_register_slice.sv",
                "linear_layers/mxint_operators/rtl/unpacked_mx_fifo.sv",
                "common/rtl/unpacked_register_slice.sv",
                "common/rtl/split2.sv",
                "common/rtl/join2.sv",
                "memory/rtl/unpacked_skid_buffer.sv",
                "memory/rtl/skid_buffer.sv",
                "memory/rtl/ultraram_fifo.sv",
                "memory/rtl/ultraram.v",
                "linear_layers/fixed_operators/rtl/fixed_dot_product.sv",
                "linear_layers/fixed_operators/rtl/fixed_vector_mult.sv",
                "linear_layers/fixed_operators/rtl/fixed_mult.sv",
                "linear_layers/fixed_operators/rtl/fixed_adder_tree.sv",
                "linear_layers/fixed_operators/rtl/fixed_adder_tree_layer.sv",
            ],
        },
    },
    "relu": {
        "fixed": {
            "name": "fixed_relu",
            "dependence_files": [
                "activation_layers/rtl/fixed_relu.sv",
            ],
        },
        "mxint": {
            "name": "mxint_relu",
            "dependence_files": [
                "linear_layers/mxint_operators/rtl/mxint_relu.sv",
                "linear_layers/mxint_operators/rtl/mxint_cast.sv",
                "common/rtl/split2.sv",
                "linear_layers/mxint_operators/rtl/log2_max_abs.sv",
                "linear_layers/mxint_operators/rtl/or_tree.sv",
                "linear_layers/mxint_operators/rtl/or_tree_layer.sv",
                "common/rtl/register_slice.sv",
                "linear_layers/mxint_operators/rtl/unpacked_mx_fifo.sv",
                "memory/rtl/fifo.sv",
                "memory/rtl/skid_buffer.sv",
                "memory/rtl/simple_dual_port_ram.sv",
                "common/rtl/join2.sv",
            ],
        },
    },
    "hardshrink": {
        "fixed": {
            "name": "fixed_hardshrink",
            "dependence_files": [
                "activation_layers/rtl/fixed_hardshrink.sv",
            ],
        },
    },
    "silu": {
        "fixed": {
            "name": "fixed_silu",
            "dependence_files": [
                "activation_layers/rtl/fixed_silu.sv",
                "activation_layers/rtl/silu_lut.sv",
            ],
        },
    },
    "elu": {
        "fixed": {
            "name": "fixed_elu",
            "dependence_files": [
                "activation_layers/rtl/fixed_elu.sv",
                "activation_layers/rtl/elu_lut.sv",
            ],
        },
    },
    "sigmoid": {
        "fixed": {
            "name": "fixed_sigmoid",
            "dependence_files": [
                "activation_layers/rtl/fixed_sigmoid.sv",
                "activation_layers/rtl/sigmoid_lut.sv",
            ],
        },
    },
    "softshrink": {
        "fixed": {
            "name": "fixed_softshrink",
            "dependence_files": [
                "activation_layers/rtl/fixed_softshrink.sv",
            ],
        },
    },
    "logsigmoid": {
        "fixed": {
            "name": "fixed_logsigmoid",
            "dependence_files": [
                "activation_layers/rtl/fixed_logsigmoid.sv",
                "activation_layers/rtl/logsigmoid_lut.sv",
            ],
        },
    },
    "softmax": {
        "fixed": {
            "name": "fixed_softmax",
            "dependence_files": [
                "activation_layers/rtl/fixed_softmax.sv",
                "activation_layers/rtl/exp_lut .sv",
            ],
        },
    },
    "batch_norm2d": {"fixed": norm},
    "layer_norm": {"fixed": norm},
    "group_norm": {"fixed": norm},
    "instance_norm2d": {"fixed": norm},
    "rms_norm": {"fixed": norm},
    "selu": {
        "fixed": {
            "name": "fixed_selu",
            "dependence_files": [
                "activation_layers/rtl/fixed_selu.sv",
            ],
        },
    },
    "tanh": {
        "fixed": {
            "name": "fixed_tanh",
            "dependence_files": [
                "activation_layers/rtl/fixed_tanh.sv",
            ],
        },
    },
    "gelu": {
        "fixed": {
            "name": "fixed_gelu",
            "dependence_files": [
                "activation_layers/rtl/fixed_gelu.sv",
                "activation_layers/rtl/gelu_lut.sv",
            ],
        },
    },
    "softsign": {
        "fixed": {
            "name": "fixed_softsign",
            "dependence_files": [
                "activation_layers/rtl/fixed_softsign.sv",
                "linear_layers/fixed_operators/rtl/fixed_mult.sv",
            ],
        },
    },
    "softplus": {
        "fixed": {
            "name": "fixed_softplus",
            "dependence_files": [
                "activation_layers/rtl/fixed_softplus.sv",
            ],
        },
    },
    "add": {
        "fixed": {
            "name": "fixed_adder",
            "dependence_files": [
                "linear_layers/fixed_operators/rtl/fixed_adder.sv",
            ],
        },
    },
    "mul": {
        "fixed": {
            "name": "fixed_elementwise_multiplier",
            "dependence_files": [
                "linear_layers/fixed_operators/rtl/fixed_vector_mult.sv",
            ],
        },
    },
    "df_split": {
        "fixed": {
            "name": "df_split",
            "dependence_files": ["common/rtl/df_split.sv", "common/rtl/split2.sv"],
        },
    },
    "getitem": {
        "fixed": {
            "name": "buffer",
            "dependence_files": [
                "memory/rtl/buffer.sv",
            ],
        },
    },
}
