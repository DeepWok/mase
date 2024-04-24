INTERNAL_RTL_DEPENDENCIES = {
    "linear": [
        "cast/rtl/fixed_cast.sv",
        "linear/rtl/fixed_linear.sv",
        "fixed_arithmetic/rtl/fixed_dot_product.sv",
        "fixed_arithmetic/rtl/fixed_accumulator.sv",
        "fixed_arithmetic/rtl/fixed_vector_mult.sv",
        "fixed_arithmetic/rtl/fixed_adder_tree.sv",
        "fixed_arithmetic/rtl/fixed_adder_tree_layer.sv",
        "fixed_arithmetic/rtl/fixed_mult.sv",
        "common/rtl/register_slice.sv",
        "common/rtl/skid_buffer.sv",
        "common/rtl/join2.sv",
        "cast/rtl/fixed_rounding.sv",
        "cast/rtl/fixed_round.sv",
    ],
    "relu": [
        "activations/rtl/fixed_relu.sv",
        "cast/rtl/fixed_rounding.sv",
        "cast/rtl/fixed_round.sv",
    ],
}
