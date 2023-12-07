# mase_op : the set of functional equivalent IPs with different design configurations.
# The first IP in each list is used by default
INTERNAL_COMP = {
    "linear": [
        {
            "name": "fixed_linear",
            "dependence_files": [
                "cast/fixed_cast.sv",
                "fixed_arith/fixed_dot_product.sv",
                "fixed_arith/fixed_vector_mult.sv",
                "fixed_arith/register_slice.sv",
                "fixed_arith/fixed_accumulator.sv",
                "fixed_arith/fixed_adder_tree.sv",
                "fixed_arith/fixed_adder_tree_layer.sv",
                "fixed_arith/fixed_mult.sv",
                "common/join2.sv",
                "linear/fixed_linear.sv",
            ],
        },
    ],
    "relu": [
        {
            "name": "fixed_relu",
            "dependence_files": [
                "activations/fixed_relu.sv",
            ],
        },
    ],
}
