# mase_op : the set of functional equivalent IPs with different design configurations.
# The first IP in each list is used by default
norm = {
    "name": "norm",
    "dependence_files": [
        "common/rtl/join2.sv",
        "common/rtl/split2.sv",
        "common/rtl/repeat_circular_buffer.sv",
        "common/rtl/skid_buffer.sv",
        "common/rtl/register_slice.sv",
        "common/rtl/simple_dual_port_ram.sv",
        "common/rtl/fifo.sv",
        "common/rtl/lut.sv",
        "cast/rtl/floor_round.sv",
        "cast/rtl/signed_clamp.sv",
        "cast/rtl/fixed_signed_cast.sv",
        "fixed_arithmetic/rtl/fixed_accumulator.sv",
        "fixed_arithmetic/rtl/fixed_adder_tree.sv",
        "fixed_arithmetic/rtl/fixed_adder_tree_layer.sv",
        "fixed_arithmetic/rtl/fixed_lut_index.sv",
        "fixed_arithmetic/rtl/fixed_nr_stage.sv",
        "fixed_arithmetic/rtl/fixed_range_augmentation.sv",
        "fixed_arithmetic/rtl/fixed_range_reduction.sv",
        "fixed_arithmetic/rtl/fixed_isqrt.sv",
        "matmul/rtl/matrix_fifo.sv",
        "matmul/rtl/matrix_flatten.sv",
        "matmul/rtl/matrix_unflatten.sv",
        "norm/rtl/channel_selection.sv",
        "norm/rtl/group_norm_2d.sv",
        "norm/rtl/rms_norm_2d.sv",
        "norm/rtl/batch_norm_2d.sv",
        "norm/rtl/norm.sv",
    ],
}

INTERNAL_COMP = {
    "linear": [
        {
            "name": "fixed_linear",
            "dependence_files": [
                "cast/rtl/fixed_cast.sv",
                "fixed_arithmetic/rtl/fixed_dot_product.sv",
                "fixed_arithmetic/rtl/fixed_vector_mult.sv",
                "fixed_arithmetic/rtl/fixed_accumulator.sv",
                "fixed_arithmetic/rtl/fixed_adder_tree.sv",
                "fixed_arithmetic/rtl/fixed_adder_tree_layer.sv",
                "fixed_arithmetic/rtl/fixed_mult.sv",
                "common/rtl/register_slice.sv",
                "common/rtl/join2.sv",
                "common/rtl/skid_buffer.sv",
                "linear/rtl/fixed_linear.sv",
            ],
        },
    ],
    "relu": [
        {
            "name": "fixed_relu",
            "dependence_files": [
                "activations/rtl/fixed_relu.sv",
            ],
        },
    ],
    "batch_norm2d": [norm],
    "layer_norm": [norm],
    "group_norm": [norm],
    "instance_norm2d": [norm],
    "rms_norm": [norm],
    "selu": [
        {
            "name": "fixed_selu",
            "dependence_files": [
                "activations/rtl/fixed_selu.sv",
            ],
        },
    ],
    "tanh": [
        {
            "name": "fixed_tanh",
            "dependence_files": [
                "activations/rtl/fixed_tanh.sv",
            ],
        },
    ],
    "gelu": [
        {
            "name": "fixed_gelu",
            "dependence_files": [
                "activations/rtl/fixed_gelu.sv",
                "arithmetic/rtl/fixed_mult.sv",
            ],
        },
    ],
    "softsign": [
        {
            "name": "fixed_softsign",
            "dependence_files": [
                "activations/rtl/fixed_softsign.sv",
                "arithmetic/rtl/fixed_mult.sv",
            ],
        },
    ],
    "softplus": [
        {
            "name": "fixed_softplus",
            "dependence_files": [
                "activations/rtl/fixed_softplus.sv",
            ],
        },
    ],
}
