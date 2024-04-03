# mase_op : the set of functional equivalent IPs with different design configurations.
# The first IP in each list is used by default
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
    "hardshrink": [
        {
            "name": "fixed_hardshrink",
            "dependence_files": [
                "activations/rtl/fixed_hardshrink.sv",
            ],
        },
    ],
    "silu": [
        {
            "name": "fixed_silu",
            "dependence_files": [
                "activations/rtl/fixed_silu.sv",
                "activations/rtl/silu_lut.sv",
            ],
        },
    ],
    "elu": [
        {
            "name": "fixed_elu",
            "dependence_files": [
                "activations/rtl/fixed_elu.sv",
                "activations/rtl/elu_lut.sv",
            ],
        },
    ],
    "sigmoid": [
        {
            "name": "fixed_sigmoid",
            "dependence_files": [
                "activations/rtl/fixed_sigmoid.sv",
                "activations/rtl/sigmoid_lut.sv",
            ],
        },
    ],
    "softshrink": [
        {
            "name": "fixed_softshrink",
            "dependence_files": [
                "activations/rtl/fixed_softshrink.sv",
            ],
        },
    ],
    "logsigmoid": [
        {
            "name": "fixed_logsigmoid",
            "dependence_files": [
                "activations/rtl/fixed_logsigmoid.sv",
                "activations/rtl/logsigmoid_lut.sv",
            ],
        },
    ],
    "softmax": [
        {
            "name": "fixed_softmax",
            "dependence_files": [
                "activations/rtl/fixed_softmax.sv",
                "activations/rtl/exp_lut .sv",
            ],
        },
    ],
}
