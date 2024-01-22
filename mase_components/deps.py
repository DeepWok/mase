"""

Contains the single source of truth for modules and their dependencies.

Entry format:
"<group>/<module>": [<group1>, <group2>, ...]

"""


MASE_HW_DEPS = {
    # TODO: Failing Test
    # "activations/fixed_relu": [],
    # TODO: Failing test: missing "z_proj" in config?
    # "attention/fixed_self_att": ["attention", "cast", "common", "conv",
    #                              "fixed_arith", "linear", "matmul", "ViT"],
    "cast/fixed_cast": [],
    "cast/fixed_rounding": ["cast"],
    "linear/fixed_linear": ["cast", "common", "fixed_arith"],
    "emitted_verilog/top": [
        "cast",
        "common",
        "fixed_arith",
        "activations",
        "emitted_verilog",
        "linear",
    ],
    "fixed_arith/fixed_mult": [],
    "fixed_arith/fixed_adder_tree_layer": [],
    "fixed_arith/fixed_accumulator": ["common"],
    "fixed_arith/fixed_adder_tree": ["fixed_arith", "common"],
    "fixed_arith/fixed_vector_mult": ["fixed_arith", "common"],
    "fixed_arith/fixed_dot_product": ["fixed_arith", "common"],
    "common/cut_data": ["common"],
    "common/wrap_data": ["common"],
    "common/skid_buffer": [],
    # TODO: Geniune test case failure
    # "common/fifo": ["common"],
    "common/input_buffer": ["common"],
    # New matrix multiplication modules
    # "common/repeat_circular_buffer": ["common"],
    # "cast/fixed_signed_cast": ["cast"],
    # "matmul/simple_matmul": ["common", "linear", "cast", "fixed_arith"],
    "matmul/fixed_matmul": ["common", "linear", "cast", "fixed_arith", "matmul"],
    # "matmul/test_chain_matmul": ["common", "linear", "cast", "fixed_arith", "matmul"],
    # "matmul/transpose": [],
    # TODO: Geniune test case failure
    # "ViT/fixed_patch_embed": [
    #     "conv",
    #     "ViT",
    #     "cast",
    #     "matmul",
    #     "linear",
    #     "attention",
    #     "common",
    #     "fixed_arith",
    # ],
    # TODO: Geniune test case failure
    # "ViT/fixed_msa": [
    #     "conv",
    #     "ViT",
    #     "cast",
    #     "matmul",
    #     "linear",
    #     "attention",
    #     "common",
    #     "fixed_arith",
    # ],
    # Now is at the rounding version, so do not test cast version matmul core anymore
    # "fixed_arith/fixed_matmul_core": ["cast", "linear", "fixed_arith", "common"],
    # TODO: Broken test, check convolution_tb.py
    # "conv/convolution": ["conv", "linear", "common", "fixed_arith"],
    # TODO: check again why not passing...
    "conv/sliding_window": ["cast", "conv", "linear", "common", "fixed_arith"],
    "conv/padding": ["cast", "conv", "linear", "common", "fixed_arith"],
    "conv/convolution": ["cast", "conv", "linear", "common", "fixed_arith"],
    "matmul/fixed_matmul": ["cast", "linear", "matmul", "common", "fixed_arith"],
    # # 'cast/bram_cast': [],
    # # 'cast/bram2hs_cast': [],
    # # 'cast/hs2bram_cast': [],
    # # 'common/ram_block': [],
    # # 'common/join2': [],
    # "binary_arith/binary_activation_binary_mult": [],
    # "binary_arith/binary_activation_binary_vector_mult": ["binary_arith", "common"],
    # "binary_arith/binary_activation_binary_adder_tree_layer": [],
    # "binary_arith/binary_activation_binary_adder_tree": ["binary_arith", "common"],
    # "binary_arith/binary_activation_binary_dot_product": ["binary_arith", "common"],
    # "binary_arith/fixed_activation_binary_mult": [],
    # "binary_arith/fixed_activation_binary_vector_mult": ["binary_arith", "common"],
    # "binary_arith/fixed_activation_binary_dot_product": [
    #     "binary_arith",
    #     "fixed_arith",
    #     "common",
    # ],
    # "linear/binary_activation_binary_linear": [
    #     "cast",
    #     "linear",
    #     "fixed_arith",
    #     "binary_arith",
    #     "common",
    # ],
    # "linear/fixed_activation_binary_linear": [
    #     "cast",
    #     "linear",
    #     "fixed_arith",
    #     "binary_arith",
    #     "common",
    # ],
    # 'activations/int_relu6': ['common'],
}
