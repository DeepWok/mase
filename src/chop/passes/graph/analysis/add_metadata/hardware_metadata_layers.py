# mase_op : the set of functional equivalent IPs with different design configurations.
# The first IP in each list is used by default
norm = {
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
linear = {
        "name": "fixed_linear",
        "dependence_files": [
            "cast/rtl/fixed_round.sv",
            "cast/rtl/fixed_rounding.sv",
            "cast/rtl/floor_round.sv",
            "cast/rtl/signed_clamp.sv",
            "cast/rtl/fixed_signed_cast.sv",
            "linear_layers/fixed_operators/rtl/fixed_dot_product.sv",
            "linear_layers/fixed_operators/rtl/fixed_vector_mult.sv",
            "linear_layers/fixed_operators/rtl/fixed_accumulator.sv",
            "linear_layers/fixed_operators/rtl/fixed_adder_tree.sv",
            "linear_layers/fixed_operators/rtl/fixed_adder_tree_layer.sv",
            "linear_layers/fixed_operators/rtl/fixed_mult.sv",
            "linear_layers/matmul/rtl/matrix_flatten.sv",
            "linear_layers/matmul/rtl/matrix_unflatten.sv",
            "linear_layers/matmul/rtl/matrix_fifo.sv",
            "linear_layers/matmul/rtl/matrix_accumulator.sv",
            "linear_layers/matmul/rtl/simple_matmul.sv",
            "linear_layers/matmul/rtl/matmul.sv",
            "linear_layers/matmul/rtl/transpose.sv",
            "linear_layers/matmul/rtl/matrix_stream_transpose.sv",
            "common/rtl/register_slice.sv",
            "common/rtl/join2.sv",
            "common/rtl/mux.sv",
            "common/rtl/unpacked_register_slice.sv",
            "common/rtl/single_element_repeat.sv",
            "memory/rtl/unpacked_repeat_circular_buffer.sv",
            "memory/rtl/skid_buffer.sv",
            "memory/rtl/input_buffer.sv",
            "memory/rtl/blk_mem_gen_0.sv",
            "memory/rtl/fifo.sv",
            "memory/rtl/simple_dual_port_ram.sv",
            "linear_layers/fixed_linear_layer/rtl/fixed_linear.sv",
        ],
}
INTERNAL_COMP = {
    "linear": [linear],
    "relu": [
        {
            "name": "fixed_relu",
            "dependence_files": [
                "activation_layers/rtl/fixed_relu.sv",
            ],
        },
    ],
    "hardshrink": [
        {
            "name": "fixed_hardshrink",
            "dependence_files": [
                "activation_layers/rtl/fixed_hardshrink.sv",
            ],
        },
    ],
    "silu": [
        {
            "name": "fixed_silu",
            "dependence_files": [
                "activation_layers/rtl/fixed_silu.sv",
                "activation_layers/rtl/silu_lut.sv",
            ],
        },
    ],
    "elu": [
        {
            "name": "fixed_elu",
            "dependence_files": [
                "activation_layers/rtl/fixed_elu.sv",
                "activation_layers/rtl/elu_lut.sv",
            ],
        },
    ],
    "sigmoid": [
        {
            "name": "fixed_sigmoid",
            "dependence_files": [
                "activation_layers/rtl/fixed_sigmoid.sv",
                "activation_layers/rtl/sigmoid_lut.sv",
            ],
        },
    ],
    "softshrink": [
        {
            "name": "fixed_softshrink",
            "dependence_files": [
                "activation_layers/rtl/fixed_softshrink.sv",
            ],
        },
    ],
    "logsigmoid": [
        {
            "name": "fixed_logsigmoid",
            "dependence_files": [
                "activation_layers/rtl/fixed_logsigmoid.sv",
                "activation_layers/rtl/logsigmoid_lut.sv",
            ],
        },
    ],
    "softmax": [
        {
            "name": "fixed_softmax",
            "dependence_files": [
                "activation_layers/rtl/fixed_softmax.sv",
                "activation_layers/rtl/exp_lut .sv",
            ],
        }
    ],
    "batch_norm2d": [norm],
    "group_norm": [norm],
    "instance_norm2d": [norm],
    "rms_norm": [norm],
    "layer_norm": [
        {
            "name": "layer_norm_2d",
            "dependence_files": norm["dependence_files"]+ [
                "normalization_layers/rtl/layer_norm_2d.sv",
                "generated_lut/rtl/isqrt_lut.sv"
            ],
        },
    ],
    "selu": [
        {
            "name": "fixed_selu",
            "dependence_files": [
                "activation_layers/rtl/fixed_selu.sv",
            ],
        },
    ],
    "tanh": [
        {
            "name": "fixed_tanh",
            "dependence_files": [
                "activation_layers/rtl/fixed_tanh.sv",
            ],
        },
    ],
    "gelu": [
        {
            "name": "fixed_gelu",
            "dependence_files": [
                "activation_layers/rtl/fixed_gelu.sv",
                "generated_lut/rtl/gelu_lut.sv",
                "common/rtl/unpacked_register_slice_quick.sv",
            ],
        },
    ],
    "softsign": [
        {
            "name": "fixed_softsign",
            "dependence_files": [
                "activation_layers/rtl/fixed_softsign.sv",
                "linear_layers/fixed_operators/rtl/fixed_mult.sv",
            ],
        },
    ],
    "softplus": [
        {
            "name": "fixed_softplus",
            "dependence_files": [
                "activation_layers/rtl/fixed_softplus.sv",
            ],
        },
    ],
    "add": [
        {
            "name": "fixed_adder",
            "dependence_files": [
                "linear_layers/fixed_operators/rtl/fixed_adder.sv",
            ],
        }
    ],
    "mul": [
        {
            "name": "fixed_elementwise_multiplier",
            "dependence_files": [
                "linear_layers/fixed_operators/rtl/fixed_vector_mult.sv",
            ],
        }
    ],
    "df_split": [
        {
            "name": "df_split",
            "dependence_files": ["common/rtl/df_split.sv", "common/rtl/split2.sv"],
        }
    ],
    "fork2": [
        {
            "name": "fork2",
            "dependence_files": ["common/rtl/fork2.sv"],
        }
    ],
    "getitem": [
        {
            "name": "buffer",
            "dependence_files": [
                "memory/rtl/buffer.sv",
            ],
        }
    ],
    "vit_self_attention_integer": [
        {
            "name": "fixed_vit_attention_single_precision_wrapper",
            "dependence_files": linear["dependence_files"] + [
                "vision_models/vit/rtl/fixed_vit_attention_single_precision_wrapper.sv",
                "vision_models/vit/rtl/fixed_vit_attention.sv",
                "vision_models/vit/rtl/fixed_vit_attention_head.sv", 
                "transformer_layers/rtl/self_attention_head_single_scatter.sv",
                "transformer_layers/rtl/gqa_head_scatter_control.sv",
                "transformer_layers/rtl/self_attention_head_gather.sv",
                "transformer_layers/rtl/fixed_self_attention_input_block_batched.sv",
                "transformer_layers/rtl/self_attention_head_scatter.sv",
                "activation_layers/rtl/fixed_softmax.sv",
                "scalar_operators/fixed/rtl/fixed_div.sv",
                "generated_lut/rtl/exp_lut.sv",
                "common/rtl/find_first_arbiter.sv",
                "common/rtl/split2.sv",
                "common/rtl/split_n.sv",
                "memory/rtl/unpacked_fifo.sv",
            ],
        }
    ],
    "grouped_query_attention": [
        {
            "name": "fixed_gqa_wrapper",
            "dependence_files": [
                "common/rtl/find_first_arbiter.sv",
                "common/rtl/mux.sv",
                "common/rtl/join2.sv",
                "common/rtl/split2.sv",
                "common/rtl/split_n.sv",
                "common/rtl/join_n.sv",
                "memory/rtl/repeat_circular_buffer.sv",
                "common/rtl/single_element_repeat.sv",
                "memory/rtl/skid_buffer.sv",
                "memory/rtl/unpacked_skid_buffer.sv",
                "common/rtl/register_slice.sv",
                "memory/rtl/simple_dual_port_ram.sv",
                "memory/rtl/fifo.sv",
                "common/rtl/lut.sv",
                "common/rtl/comparator_tree.sv",
                "common/rtl/comparator_accumulator.sv",
                "common/rtl/unpacked_register_slice.sv",
                "cast/rtl/fixed_round.sv",
                "cast/rtl/fixed_rounding.sv",
                "cast/rtl/floor_round.sv",
                "cast/rtl/signed_clamp.sv",
                "cast/rtl/fixed_signed_cast.sv",
                "linear_layers/fixed_operators/rtl/fixed_mult.sv",
                "linear_layers/fixed_operators/rtl/fixed_vector_mult.sv",
                "linear_layers/fixed_operators/rtl/fixed_dot_product.sv",
                "linear_layers/fixed_operators/rtl/fixed_accumulator.sv",
                "linear_layers/fixed_operators/rtl/fixed_adder_tree_layer.sv",
                "linear_layers/fixed_operators/rtl/fixed_adder_tree.sv",
                "linear_layers/fixed_operators/rtl/fixed_range_reduction.sv",
                "linear_layers/fixed_linear_layer/rtl/fixed_linear.sv",
                "linear_layers/matmul/rtl/matrix_flatten.sv",
                "linear_layers/matmul/rtl/matrix_unflatten.sv",
                "linear_layers/matmul/rtl/matrix_fifo.sv",
                "linear_layers/matmul/rtl/matrix_accumulator.sv",
                "linear_layers/matmul/rtl/simple_matmul.sv",
                "linear_layers/matmul/rtl/matmul.sv",
                "linear_layers/matmul/rtl/transpose.sv",
                "linear_layers/matmul/rtl/matrix_stream_transpose.sv",
                "activation_layers/rtl/softermax_lpw_pow2.sv",
                "activation_layers/rtl/softermax_lpw_reciprocal.sv",
                "activation_layers/rtl/softermax_local_window.sv",
                "activation_layers/rtl/softermax_global_norm.sv",
                "activation_layers/rtl/fixed_softermax_1d.sv",
                "activation_layers/rtl/fixed_softermax.sv",
                "transformer_layers/rtl/fixed_gqa_projections.sv",
                "transformer_layers/rtl/self_attention_head_single_scatter.sv",
                "transformer_layers/rtl/gqa_head_scatter_control.sv",
                "transformer_layers/rtl/self_attention_head_gather.sv",
                "transformer_layers/rtl/fixed_self_attention_head.sv",
                "transformer_layers/rtl/fixed_grouped_query_attention.sv",
                "transformer_layers/rtl/fixed_gqa_wrapper.sv",
            ],
        }
    ],
}
