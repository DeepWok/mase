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
    "batch_norm1d": [
        {
            "name": "fixed_batch_norm1d",
            "dependence_files": ["normalisation/rtl/fixed_batch_norm1d.sv"],
        },
    ],
    "layer_norm": [
        {
            "name": "fixed_layer_norm",
            "dependence_files": ["normalisation/rtl/fixed_layer_norm.sv"],
        },
    ],
}
