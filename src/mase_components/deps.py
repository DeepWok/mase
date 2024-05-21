"""

Contains the single source of truth for modules and their dependencies.

Entry format:
"<group>/<module>": [<group1>, <group2>, ...]

"""

MASE_HW_DEPS = {
    # Activations
    "activations/fixed_relu": [],
    "activations/fixed_leaky_relu": [],
    "activations/fixed_tanh": ["cast"],
    "activations/fixed_selu": ["cast", "activations", "fixed_math"],
    "activations/fixed_gelu": ["fixed_arithmetic", "common"],
    "activations/fixed_softsign": ["fixed_arithmetic", "common"],
    "activations/fixed_softplus": [],
    "activations/fixed_hardshrink": ["common", "cast"],
    "activations/fixed_hardswish": ["common", "fixed_arithmetic"],
    "activations/fixed_silu": ["common", "cast", "activations"],
    "activations/fixed_elu": ["common", "cast", "activations"],
    "activations/fixed_sigmoid": ["common", "cast", "activations"],
    "activations/fixed_softshrink": ["common", "cast"],
    "activations/fixed_logsigmoid": ["common", "cast", "activations"],
    "activations/fixed_softmax": [
        "common",
        "cast",
        "fixed_arithmetic",
        "conv",
        "activations",
    ],
    "activations/fixed_softermax_1d": [
        "common",
        "cast",
        "fixed_arithmetic",
        "conv",
        "matmul",
        "activations",
    ],
    # Attention
    "attention/fixed_self_attention": [
        "activations",
        "arbiters",
        "attention",
        "cast",
        "common",
        "fixed_arithmetic",
        "linear",
        "matmul",
    ],
    "attention/fixed_self_attention_head": [
        "attention",
        "cast",
        "common",
        "fixed_arithmetic",
        "linear",
        "matmul",
        "activations",
    ],
    "attention/fixed_grouped_query_attention": [
        "attention",
        "arbiters",
        "cast",
        "common",
        "fixed_arithmetic",
        "linear",
        "matmul",
        "activations",
    ],
    "arithmetic/mac": ["fixed_arithmetic", "float_arithmetic"],
    # Binary arithmetic
    "binary_arith/binary_activation_binary_mult": [],
    "binary_arith/binary_activation_binary_vector_mult": ["binary_arith", "common"],
    "binary_arith/binary_activation_binary_adder_tree_layer": [],
    "binary_arith/binary_activation_binary_adder_tree": ["binary_arith", "common"],
    "binary_arith/binary_activation_binary_dot_product": ["binary_arith", "common"],
    "binary_arith/fixed_activation_binary_mult": [],
    "binary_arith/fixed_activation_binary_vector_mult": ["binary_arith", "common"],
    "binary_arith/fixed_activation_binary_dot_product": [
        "binary_arith",
        "fixed_arithmetic",
        "common",
    ],
    "buffers/hybrid_buffer": ["buffers"],
    # Linear
    "linear/fixed_linear": ["matmul", "cast", "common", "fixed_arithmetic"],
    "linear/binary_activation_binary_linear": [
        "cast",
        "linear",
        "fixed_arithmetic",
        "binary_arith",
        "common",
    ],
    "linear/fixed_activation_binary_linear": [
        "cast",
        "linear",
        "fixed_arithmetic",
        "binary_arith",
        "common",
    ],
    # Fixed arithmetic
    "fixed_arithmetic/fixed_range_reduction": [],
    "fixed_arithmetic/fixed_lut_index": [],
    "fixed_arithmetic/fixed_range_augmentation": [],
    "fixed_arithmetic/fixed_mult": [],
    "fixed_arithmetic/fixed_accumulator": ["common"],
    "fixed_arithmetic/fixed_adder_tree": ["fixed_arithmetic", "common"],
    "fixed_arithmetic/fixed_vector_mult": ["fixed_arithmetic", "common"],
    "fixed_arithmetic/fixed_dot_product": ["fixed_arithmetic", "common"],
    "fixed_arithmetic/fixed_matmul_core": [
        "fixed_arithmetic",
        "common",
        "linear",
        "cast",
        "matmul",
    ],
    # Fixed math
    "fixed_math/fixed_exp": ["fixed_math", "cast"],
    "fixed_math/fixed_isqrt": ["fixed_math", "fixed_arithmetic", "common"],
    "fixed_math/fixed_nr_stage": ["fixed_math", "common"],
    "fixed_math/fixed_series_approx": ["fixed_math"],
    # Float arithmetic
    "float_arithmetic/float_mac": ["float_arithmetic"],
    "float_arithmetic/float_multiplier": ["float_arithmetic"],
    # Cast
    "cast/fixed_cast": [],
    "cast/fixed_rounding": ["cast"],
    "cast/fixed_signed_cast": ["cast"],
    "cast/bram_cast": ["memory"],
    "cast/bram2hs_cast": ["memory"],
    "cast/hs2bram_cast": ["memory"],
    # Common
    "common/cut_data": ["common"],
    "common/wrap_data": ["common"],
    "common/skid_buffer": [],
    "common/fifo": ["common"],
    "common/input_buffer": ["common"],
    "common/repeat_circular_buffer": ["common"],
    "common/lut": [],
    "common/ram_block": [],
    "common/join2": [],
    "common/register_slice": ["common"],
    "common/unpacked_fifo": ["common"],
    "common/unpacked_skid_buffer": ["common"],
    # Convolution
    "conv/convolution": [
        "cast",
        "conv",
        "linear",
        "common",
        "fixed_arithmetic",
        "matmul",
    ],
    "conv/binary_activation_binary_convolution": [
        "cast",
        "conv",
        "linear",
        "common",
        "fixed_arithmetic",
    ],
    "conv/sliding_window": ["cast", "conv", "linear", "common", "fixed_arithmetic"],
    "conv/padding": ["cast", "conv", "linear", "common", "fixed_arithmetic"],
    # Matmul
    "matmul/simple_matmul": ["common", "linear", "cast", "fixed_arithmetic", "matmul"],
    "matmul/fixed_matmul": ["common", "linear", "cast", "fixed_arithmetic", "matmul"],
    "matmul/matmul": ["common", "linear", "cast", "fixed_arithmetic", "matmul"],
    "matmul/test_chain_matmul": [
        "common",
        "linear",
        "cast",
        "fixed_arithmetic",
        "matmul",
    ],
    "matmul/transpose": [],
    "matmul/matrix_stream_transpose": ["common", "matmul"],
    # Norm
    "norm/group_norm_2d": ["common", "matmul", "fixed_arithmetic", "norm", "cast"],
    "norm/rms_norm_2d": ["common", "matmul", "fixed_arithmetic", "norm", "cast"],
    "norm/batch_norm_2d": ["norm", "common", "cast", "matmul"],
    # LLM int8
    "llm/scatter": ["llm"],
    "llm/dequantizer": ["llm", "common", "cast", "fixed_arithmetic"],
    "llm/fixed_comparator_tree_layer": ["llm"],
    "llm/fixed_comparator_tree": ["llm", "common"],
    "llm/fixed_linear_dequant": ["llm", "common", "cast", "fixed_arithmetic"],
    "llm/fixed_matmul_core_dequant": ["llm", "common", "cast", "fixed_arithmetic"],
    "llm/quantizer_top": ["llm", "cast", "common", "fixed_arithmetic"],
    "llm/quantizer_part": ["llm", "cast", "common", "fixed_arithmetic"],
    "llm/find_max": ["llm", "common"],
    "llm/quantized_matmul": [
        "llm",
        "fixed_arithmetic",
        "cast",
        "linear",
        "matmul",
        "common",
    ],
    "llm/llm_int8_top": [
        "llm",
        "fixed_arithmetic",
        "cast",
        "linear",
        "matmul",
        "common",
    ],
    # ViT
    "ViT/fixed_patch_embed": [
        "conv",
        "ViT",
        "cast",
        "matmul",
        "linear",
        "attention",
        "common",
        "fixed_arithmetic",
    ],
    "ViT/fixed_msa": [
        "conv",
        "ViT",
        "cast",
        "matmul",
        "linear",
        "attention",
        "common",
        "fixed_arithmetic",
    ],
}
