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
    #                              "fixed_arithmetic", "linear", "matmul", "ViT"],
    "activations/softermax_lpw_pow2": ["common", "cast"],
    "cast/fixed_cast": [],
    "cast/fixed_rounding": ["cast"],
    "linear/fixed_linear": ["cast", "common", "fixed_arithmetic"],
    # Commented out ISQRT sub-components to shorten CI runs:
    # "fixed_arithmetic/fixed_nr_stage": ["common"],
    # "fixed_arithmetic/fixed_range_reduction": [],
    # "fixed_arithmetic/fixed_lut_index": [],
    # "fixed_arithmetic/fixed_range_augmentation": [],
    "fixed_arithmetic/fixed_isqrt": ["common", "fixed_arithmetic"],
    "fixed_arithmetic/fixed_mult": [],
    "fixed_arithmetic/fixed_adder_tree_layer": [],
    "fixed_arithmetic/fixed_accumulator": ["common"],
    "fixed_arithmetic/fixed_adder_tree": ["fixed_arithmetic", "common"],
    "fixed_arithmetic/fixed_vector_mult": ["fixed_arithmetic", "common"],
    "fixed_arithmetic/fixed_dot_product": ["fixed_arithmetic", "common"],
    "common/cut_data": ["common"],
    "common/wrap_data": ["common"],
    "common/skid_buffer": [],
    # TODO: Geniune test case failure
    "common/fifo": ["common"],
    "common/input_buffer": ["common"],
    # New matrix multiplication modules
    "common/repeat_circular_buffer": ["common"],
    "common/lut": [],
    "common/comparator_tree": ["common"],
    "common/comparator_accumulator": ["common"],
    "cast/fixed_signed_cast": ["cast"],
    "matmul/simple_matmul": ["common", "linear", "cast", "fixed_arithmetic"],
    # "matmul/fixed_matmul": ["common", "linear", "cast", "fixed_arithmetic", "matmul"],
    "matmul/test_chain_matmul": [
        "common",
        "linear",
        "cast",
        "fixed_arithmetic",
        "matmul",
    ],
    "matmul/transpose": [],
    "matmul/matrix_stream_transpose": ["common", "matmul"],
    "norm/group_norm_2d": ["common", "matmul", "fixed_arithmetic", "norm", "cast"],
    "norm/rms_norm_2d": ["common", "matmul", "fixed_arithmetic", "norm", "cast"],
    "norm/batch_norm_2d": ["norm", "common", "cast", "matmul"],
    # TODO: Geniune test case failure
    # "ViT/fixed_patch_embed": [
    #     "conv",
    #     "ViT",
    #     "cast",
    #     "matmul",
    #     "linear",
    #     "attention",
    #     "common",
    #     "fixed_arithmetic",
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
    #     "fixed_arithmetic",
    # ],
    # Now is at the rounding version, so do not test cast version matmul core anymore
    # "fixed_arithmetic/fixed_matmul_core": ["cast", "linear", "fixed_arithmetic", "common"],
    # TODO: Broken test, check convolution_tb.py
    # "conv/convolution": ["conv", "linear", "common", "fixed_arithmetic"],
    # TODO: check again why not passing...
    "conv/sliding_window": ["cast", "conv", "linear", "common", "fixed_arithmetic"],
    "conv/padding": ["cast", "conv", "linear", "common", "fixed_arithmetic"],
    # "conv/convolution": ["cast", "conv", "linear", "common", "fixed_arithmetic"],
    # "matmul/fixed_matmul": ["cast", "linear", "matmul", "common", "fixed_arithmetic"],
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
    #     "fixed_arithmetic",
    #     "common",
    # ],
    # "linear/binary_activation_binary_linear": [
    #     "cast",
    #     "linear",
    #     "fixed_arithmetic",
    #     "binary_arith",
    #     "common",
    # ],
    # "linear/fixed_activation_binary_linear": [
    #     "cast",
    #     "linear",
    #     "fixed_arithmetic",
    #     "binary_arith",
    #     "common",
    # ],
    # 'activations/int_relu6': ['common'],
}
