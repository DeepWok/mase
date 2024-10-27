"""

Contains the single source of truth for modules and their dependencies.

Entry format:
"<group>/<module>": [<group1>, <group2>, ...]

"""

MASE_HW_DEPS = {
    # Activations
    "activation_layers/fixed_relu": [],
    "activation_layers/fixed_leaky_relu": [],
    "activation_layers/fixed_tanh": ["cast"],
    "activation_layers/fixed_selu": [
        "cast",
        "activation_layers",
        "scalar_operators/fixed",
    ],
    "activation_layers/fixed_gelu": [
        "common",
        "memory",
        "activation_layers",
        "generated_lut",
    ],
    "activation_layers/fixed_softsign": [
        "common",
        "activation_layers",
        "linear_layers/fixed_operators",
    ],
    "activation_layers/fixed_softplus": ["activation_layers"],
    "activation_layers/fixed_hardshrink": ["common", "cast"],
    "activation_layers/fixed_hardswish": ["common", "fixed_arithmetic"],
    "activation_layers/fixed_silu": ["common", "cast", "activation_layers"],
    "activation_layers/fixed_elu": ["common", "cast", "activation_layers"],
    "activation_layers/fixed_sigmoid": ["common", "cast", "activation_layers"],
    "activation_layers/fixed_softshrink": ["common", "cast"],
    "activation_layers/fixed_logsigmoid": ["common", "cast", "activation_layers"],
    "activation_layers/fixed_softmax": [
        "common",
        "memory",
        "scalar_operators/fixed",
        "cast",
        "linear_layers/fixed_operators",
        "generated_lut",
        "activation_layers",
    ],
    "activation_layers/fixed_softermax_1d": [
        "common",
        "cast",
        "fixed_arithmetic",
        "conv",
        "matmul",
        "memory",
        "linear_layers/fixed_operators",
        "linear_layers/matmul",
        "activation_layers",
    ],
    # Cast
    "cast/fixed_cast": [],
    "cast/fixed_rounding": ["cast"],
    "cast/fixed_signed_cast": ["cast"],
    "cast/fixed_unsigned_cast": ["cast"],
    "cast/bram_cast": ["memory"],
    "cast/bram2hs_cast": ["memory"],
    "cast/hs2bram_cast": ["memory"],
    # Common
    "common/comparator_accumulator": ["common", "memory"],
    "common/comparator_tree": ["common", "memory"],
    "common/cut_data": ["common", "memory"],
    "common/lut": [],
    "common/register_slice": ["common"],
    "common/single_element_repeat": ["memory"],
    "common/wrap_data": ["common"],
    "common/join2": [],
    # Convolution
    "convolution_layers/convolution": [
        "cast",
        "convolution_layers",
        "common",
        "memory",
        "linear_layers/fixed_linear_layer",
        "linear_layers/fixed_operators",
        "linear_layers/matmul",
    ],
    "convolution_layers/binary_activation_layer_binary_convolution": [
        "cast",
        "conv",
        "linear",
        "common",
        "fixed_arithmetic",
    ],
    "conv/sliding_window": ["cast", "conv", "linear", "common", "fixed_arithmetic"],
    "conv/padding": ["cast", "conv", "linear", "common", "fixed_arithmetic"],
    # Language models llmint8
    "language_models/llmint8/find_max": [
        "language_models/llmint8",
        "memory",
        "common",
    ],
    "language_models/llmint8/fixed_comparator_tree_layer": [
        "language_models/llmint8",
        "memory",
        "common",
    ],
    "language_models/llmint8/fixed_comparator_tree": [
        "language_models/llmint8",
        "memory",
        "common",
    ],
    "language_models/llmint8/llm_int8_top": [
        "language_models/llmint8",
        "linear_layers/fixed_operators",
        "linear_layers/fixed_linear_layer",
        "memory",
        "common",
        "cast",
    ],
    "language_models/llmint8/quantized_matmul": [
        "language_models/llmint8",
        "linear_layers/fixed_operators",
        "memory",
        "common",
        "cast",
    ],
    "language_models/llmint8/quantizer_top": [
        "language_models/llmint8",
        "linear_layers/fixed_operators",
        "memory",
        "common",
        "cast",
    ],
    "language_models/llmint8/scatter": [
        "language_models/llmint8",
        "memory",
        "common",
    ],
    # Linear
    "linear_layers/fixed_linear_layer/fixed_linear": [
        "cast",
        "common",
        "memory",
        "linear_layers/matmul",
        "linear_layers/fixed_operators",
        "scalar_operators/fixed",
    ],
    "linear_layers/fixed_linear_layer/binary_activation_binary_linear": [
        "cast",
        "linear_layers/fixed_linear_layer",
        "linear_layers/fixed_operators",
        "linear_layers/binarized_operators",
        "common",
        "memory",
    ],
    "linear_layers/fixed_linear_layer/fixed_activation_binary_linear": [
        "cast",
        "linear_layers/fixed_linear_layer",
        "linear_layers/fixed_operators",
        "linear_layers/binarized_operators",
        "common",
        "memory",
    ],
    # Linear/Binary arithmetic
    "linear_layers/binarized_operators/binary_activation_layer_binary_mult": [],
    "linear_layers/binarized_operators/binary_activation_layer_binary_vector_mult": [
        "linear_layers/binarized_operators",
        "common",
    ],
    "linear_layers/binarized_operators/binary_activation_binary_adder_tree_layer": [],
    "linear_layers/binarized_operators/binary_activation_layer_binary_adder_tree": [
        "linear_layers/binarized_operators",
        "common",
    ],
    "linear_layers/binarized_operators/binary_activation_layer_binary_dot_product": [
        "linear_layers/binarized_operators",
        "common",
    ],
    "linear_layers/binarized_operators/fixed_activation_layer_binary_mult": [],
    "linear_layers/binarized_operators/fixed_activation_layer_binary_vector_mult": [
        "linear_layers/binarized_operators",
        "common",
    ],
    "linear_layers/binarized_operators/fixed_activation_layer_binary_dot_product": [
        "linear_layers/binarized_operators",
        "fixed_arithmetic",
        "common",
    ],
    "linear_layers/binarized_operators/hybrid_buffer": ["buffers"],
    # Linear/Fixed-point arithmetic
    "linear_layers/fixed_operators/fixed_accumulator": ["common", "memory"],
    "linear_layers/fixed_operators/fixed_adder_tree": [
        "linear_layers/fixed_operators",
        "common",
        "memory",
    ],
    "linear_layers/fixed_operators/fixed_adder_tree_layer": ["common", "memory"],
    "linear_layers/fixed_operators/fixed_dot_product": [
        "linear_layers/fixed_operators",
        "common",
        "memory",
    ],
    # TODO: continue here
    "linear_layers/fixed_operators/fixed_isqrt": [
        "linear_layers/fixed_operators",
        "common",
        "memory",
    ],
    "linear_layers/fixed_operators/fixed_range_reduction": [],
    "linear_layers/fixed_operators/fixed_lut_index": [],
    "linear_layers/fixed_operators/fixed_range_augmentation": [],
    "linear_layers/fixed_operators/fixed_mult": [],
    "linear_layers/fixed_operators/fixed_vector_mult": ["fixed_arithmetic", "common"],
    "linear_layers/fixed_operators/fixed_matmul_core": [
        "fixed_arithmetic",
        "common",
        "linear_layers/fixed_linear_layer",
        "cast",
    ],
    "linear_layers/matmul/chain_matmul": [
        "linear_layers/matmul",
        "linear_layers/fixed_linear_layer",
        "linear_layers/fixed_operators",
        "cast",
        "common",
        "memory",
    ],
    "linear_layers/matmul/fixed_matmul": [
        "linear_layers/matmul",
        "linear_layers/fixed_operators",
        "memory",
    ],
    "linear_layers/matmul/matmul": [
        "linear_layers/matmul",
        "linear_layers/fixed_operators",
        "cast",
        "memory",
        "common",
    ],
    "linear_layers/matmul/matrix_stream_transpose": [
        "linear_layers/matmul",
        "linear_layers/fixed_operators",
        "cast",
        "memory",
        "common",
    ],
    "linear_layers/matmul/simple_matmul": [
        "linear_layers/matmul",
        "linear_layers/fixed_operators",
        "cast",
        "common",
        "memory",
    ],
    "linear_layers/matmul/transpose": ["cast", "common", "memory"],
    "linear_layers/mxint_operators/mxint_vector_mult": [
        "linear_layers/mxint_operators",
        "common",
        "memory",
    ],
    "linear_layers/mxint_operators/mxint_accumulator": [
        "linear_layers/mxint_operators",
        "linear_layers/fixed_operators",
        "common",
        "memory",
    ],
    "linear_layers/mxint_operators/mxint_dot_product": [
        "linear_layers/mxint_operators",
        "linear_layers/fixed_operators",
        "common",
        "memory",
    ],
    "linear_layers/mxint_operators/mxint_linear": [
        "linear_layers/mxint_operators",
        "linear_layers/fixed_operators",
        "common",
        "memory",
        "cast",
    ],
    "linear_layers/mxint_operators/mxint_cast": [
        "linear_layers/mxint_operators",
        "linear_layers/fixed_operators",
        "common",
        "memory",
        "cast",
    ],
    "linear_layers/mxint_operators/old_linear": [
        "linear_layers/mxint_operators",
        "linear_layers/fixed_operators",
        "common",
        "memory",
        "cast",
    ],
    "linear_layers/mxint_operators/mxint_linear": [
        "linear_layers/fixed_linear_layer",
        "linear_layers/matmul",
        "linear_layers/mxint_operators",
        "linear_layers/fixed_operators",
        "common",
        "memory",
        "cast",
    ],
    "linear_layers/mxint_operators/mxint_matmul": [
        "linear_layers/matmul",
        "linear_layers/mxint_operators",
        "linear_layers/fixed_operators",
        "common",
        "memory",
        "cast",
    ],
    "linear_layers/mxint_operators/log2_max_abs": [
        "linear_layers/mxint_operators",
        "common",
        "memory",
    ],
    # Memory
    "memory/skid_buffer": [],
    "memory/fifo": ["memory"],
    "memory/input_buffer": ["memory"],
    "memory/repeat_circular_buffer": ["memory"],
    "memory/ram_block": [],
    "memory/unpacked_fifo": ["memory"],
    "memory/unpacked_skid_buffer": ["memory"],
    # Normalization Layers
    "normalization_layers/batch_norm_2d": [
        "normalization_layers",
        "common",
        "cast",
        "linear_layers/matmul",
        "memory",
    ],
    "normalization_layers/channel_selection": [],
    "normalization_layers/group_norm_2d": [
        "common",
        "linear_layers/matmul",
        "linear_layers/fixed_operators",
        "normalization_layers",
        "cast",
        "memory",
        "scalar_operators/fixed",
    ],
    "normalization_layers/rms_norm_2d": [
        "common",
        "linear_layers/matmul",
        "linear_layers/fixed_operators",
        "scalar_operators/fixed",
        "normalization_layers",
        "cast",
        "memory",
    ],
    # Scalar Operators
    "scalar_operators/fixed/fixed_isqrt": [
        "memory",
        "common",
        "scalar_operators/fixed",
        "linear_layers/fixed_operators",
    ],
    "scalar_operators/fixed/fixed_nr_stage": [
        "memory",
        "common",
        "scalar_operators/fixed",
        "linear_layers/fixed_operators",
    ],
    # Transformer Layers
    "transformer_layers/fixed_self_attention": [
        "transformer_layers",
        "activation_layers",
        "arbiters",
        "cast",
        "memory",
        "common",
        "linear_layers/fixed_operators",
        "linear_layers/fixed_linear_layer",
        "linear_layers/matmul",
    ],
    "transformer_layers/fixed_self_attention_head": [
        "transformer_layers",
        "cast",
        "memory",
        "common",
        "linear_layers/fixed_operators",
        "linear_layers/fixed_linear_layer",
        "linear_layers/matmul",
        "activation_layers",
    ],
    "transformer_layers/fixed_self_attention_single_precision_wrapper": [
        "transformer_layers",
        "activation_layers",
        "arbiters",
        "cast",
        "common",
        "linear_layers/fixed_operators",
        "linear_layers/fixed_linear_layer",
    ],
    "transformer_layers/fixed_grouped_query_attention_wrapper": [
        "transformer_layers",
        "cast",
        "memory",
        "common",
        "linear_layers/fixed_operators",
        "linear_layers/fixed_linear_layer",
        "linear_layers/matmul",
        "activation_layers",
    ],
    "arithmetic/mac": ["fixed_arithmetic", "float_arithmetic"],
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
