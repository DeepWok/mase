alias ts := test-sw
alias th := test-hw
alias re := reformat

test-sw:
	bash scripts/test-machop.sh
	pytest --log-level=DEBUG --verbose \
		-n 1 \
		--cov=src/chop/ --cov-report=html \
		--html=report.html --self-contained-html \
		--junitxml=test/report.xml \
		--profile --profile-svg \
		test/

test-hw:
	# Activation_layers
	# time python3 scripts/build-components.py
	time python3 src/mase_components/activation_layers/test/fixed_gelu_tb.py
	time python3 src/mase_components/activation_layers/test/fixed_leaky_relu_tb.py
	time python3 src/mase_components/activation_layers/test/fixed_relu_tb.py
	time python3 src/mase_components/activation_layers/test/fixed_selu_tb.py
	# time python3 src/mase_components/activation_layers/test/fixed_sigmoid_tb.py
	# time python3 src/mase_components/activation_layers/test/fixed_softermax_1d_tb.py
	# time python3 src/mase_components/activation_layers/test/fixed_softermax_tb.py
	# time python3 src/mase_components/activation_layers/test/fixed_softmax_tb.py
	time python3 src/mase_components/activation_layers/test/fixed_softplus_tb.py
	time python3 src/mase_components/activation_layers/test/fixed_softsign_tb.py
	time python3 src/mase_components/activation_layers/test/fixed_tanh_tb.py
	# time python3 src/mase_components/activation_layers/test/softermax_global_norm_tb.py
	# time python3 src/mase_components/activation_layers/test/softermax_local_window_tb.py
	# time python3 src/mase_components/activation_layers/test/softermax_lpw_pow2_tb.py
	# time python3 src/mase_components/activation_layers/test/softermax_lpw_reciprocal_tb.py
	# time python3 src/mase_components/activation_layers/test/test_lint_activation_layers.py
	# time python3 src/mase_components/activation_layers/test/test_synth_activation_layers.py
	# DEV mode (no intention to fix)
	# time python3 src/mase_components/activation_layers/test/fixed_elu_tb.py
	# time python3 src/mase_components/activation_layers/test/fixed_hardshrink_tb.py
	# time python3 src/mase_components/activation_layers/test/fixed_hardswish_tb.py
	# time python3 src/mase_components/activation_layers/test/fixed_logsigmoid_tb.py
	# time python3 src/mase_components/activation_layers/test/fixed_silu_tb.py
	# time python3 src/mase_components/activation_layers/test/fixed_softshrink_tb.py

	# Cast
	time python3 src/mase_components/cast/test/fixed_cast_tb.py
	time python3 src/mase_components/cast/test/fixed_rounding_tb.py
	time python3 src/mase_components/cast/test/fixed_signed_cast_tb.py
	# time python3 src/mase_components/cast/test/fixed_unsigned_cast_tb.py

	# Common
	time python3 src/mase_components/common/test/comparator_accumulator_tb.py
	time python3 src/mase_components/common/test/cut_data_tb.py
	time python3 src/mase_components/common/test/lut_tb.py
	time python3 src/mase_components/common/test/wrap_data_tb.py
	# time python3 src/mase_components/common/test/register_slice_tb.py
	# time python3 src/mase_components/common/test/test_lint_common.py
	# DEV
	# time python3 src/mase_components/common/test/comparator_tree_tb.py
	# time python3 src/mase_components/common/test/single_element_repeat_tb.py

	# Convolution_layers
	time python3	src/mase_components/convolution_layers/test/convolution_tb.py

	# Inteface
	time python3 src/mase_components/interface/axi/test/test_lint_axi.py
	# time python3 src/mase_components/interface/axi/test/test_synth_axi.py

	# Language models llmint8
	time python3 src/mase_components/language_models/llmint8/test/find_max_tb.py
	time python3 src/mase_components/language_models/llmint8/test/fixed_comparator_tree_layer_tb.py
	time python3 src/mase_components/language_models/llmint8/test/fixed_comparator_tree_tb.py
	time python3 src/mase_components/language_models/llmint8/test/quantized_matmul_tb.py
	time python3 src/mase_components/language_models/llmint8/test/quantizer_top_tb.py
	time python3 src/mase_components/language_models/llmint8/test/scatter_tb.py
	# DEV
	# time python3 src/mase_components/language_models/llmint8/test/llm_int8_top_tb.py

	# Linear layers
	# Linear Layer - fixed_linear_layer DEBUG: use bias causes crash
	time python3	src/mase_components/linear_layers/fixed_linear_layer/test/fixed_linear_tb.py
	# time python3 src/mase_components/linear_layers/fixed_linear_layer/test/binary_activation_binary_linear_tb.py
	# time python3 src/mase_components/linear_layers/fixed_linear_layer/test/fixed_activation_binary_linear_tb.py
	# Linear Layer - fixed_operators
	time python3 src/mase_components/linear_layers/fixed_operators/test/fixed_accumulator_tb.py
	# time python3 src/mase_components/linear_layers/fixed_operators/test/fixed_adder_tree_layer_tb.py
	time python3 src/mase_components/linear_layers/fixed_operators/test/fixed_adder_tree_tb.py
	time python3 src/mase_components/linear_layers/fixed_operators/test/fixed_dot_product_tb.py
	time python3 src/mase_components/linear_layers/fixed_operators/test/fixed_lut_index_tb.py
	# time python3 src/mase_components/linear_layers/fixed_operators/test/fixed_matmul_core_tb.py
	time python3 src/mase_components/linear_layers/fixed_operators/test/fixed_mult_tb.py
	time python3 src/mase_components/linear_layers/fixed_operators/test/fixed_range_augmentation_tb.py
	# time python3 src/mase_components/linear_layers/fixed_operators/test/fixed_range_reduction_tb.py
	# Linear Layer - matmul
	# time python3 src/mase_components/linear_layers/matmul/test/chain_matmul_tb.py
	# time python3 src/mase_components/linear_layers/matmul/test/fixed_mamul_tb.py
	# time python3 src/mase_components/linear_layers/matmul/test/matmul_tb.py
	# time python3 src/mase_components/linear_layers/matmul/test/matrix_stream_transpose_tb.py
	# time python3 src/mase_components/linear_layers/matmul/test/transpose_tb.py
	# DEV Linear Layer - binary_operators
	time python3 src/mase_components/linear_layers/binarized_operators/test/binary_activation_binary_adder_tree_layer_tb.py
	# time python3 src/mase_components/linear_layers/binarized_operators/test/binary_activation_binary_adder_tree_tb.py
	# time python3 src/mase_components/linear_layers/binarized_operators/test/binary_activation_binary_dot_product_tb.py
	# time python3 src/mase_components/linear_layers/binarized_operators/test/binary_activation_binary_matmul_core_tb.py
	# time python3 src/mase_components/linear_layers/binarized_operators/test/binary_activation_binary_mult_tb.py
	# time python3 src/mase_components/linear_layers/binarized_operators/test/binary_activation_binary_vector_mult_tb.py
	# time python3 src/mase_components/linear_layers/binarized_operators/test/fixed_activation_binary_dot_product_tb.py
	# time python3 src/mase_components/linear_layers/binarized_operators/test/fixed_activation_binary_mult_tb.py
	# time python3 src/mase_components/linear_layers/binarized_operators/test/fixed_activation_binary_vector_mult_tb.py
	# time python3 src/mase_components/linear_layers/binarized_operators/test/test_lint_binary_arith.py

	# Memory
	time python3 src/mase_components/memory/test/fifo_tb.py
	# time python3 src/mase_components/memory/test/input_buffer_tb.py
	time python3 src/mase_components/memory/test/skid_buffer_tb.py
	# time python3 src/mase_components/memory/test/unpacked_fifo_tb.py
	# time python3 src/mase_components/memory/test/repeat_circular_buffer_tb.py
	# time python3 src/mase_components/memory/test/test_lint_memory.py

	# Normalization_layers
	time python3 src/mase_components/normalization_layers/test/batch_norm_2d_tb.py
	time python3 src/mase_components/normalization_layers/test/group_norm_2d_tb.py
	# DEV
	# time python3 src/mase_components/normalization_layers/test/channel_selection_tb.py
	# time python3 src/mase_components/normalization_layers/test/rms_norm_2d_tb.py
	# time python3 src/mase_components/normalization_layers/test/test_lint_norm.py

	# Scalar operators 
	time python3 src/mase_components/scalar_operators/fixed/test/fixed_isqrt_tb.py
	time python3 src/mase_components/scalar_operators/fixed/test/isqrt_sw.py
	# time python3 src/mase_components/scalar_operators/float/test/test_lint_float_arithmetic.py
	# time python3 src/mase_components/scalar_operators/fixed/test/fixed_nr_stage_tb.py
	# time python3 src/mase_components/scalar_operators/fixed/test/test_lint_fixed_math.py
	
	# Systolic array
	# time python3 src/mase_components/systolic_arrays/test/test_lint_systolic_arrays.py

	# Transformer_layers
	time python3 src/mase_components/transformer_layers/test/fixed_self_attention_head_tb.py
	# time python3 src/mase_components/transformer_layers/test/fixed_gqa_head_tb.py
	# time python3 src/mase_components/transformer_layers/test/fixed_self_attention_tb.py
	# time python3 src/mase_components/transformer_layers/test/test_lint_attention.py

reformat:
	# format python files
	black src/chop
	black src/mase_components
	black src/mase_cocotb
	black test
	# format verilog
	# find src/mase_components -name '*.sv' -exec verible-verilog-format --inplace {} +;
