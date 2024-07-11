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
	# activation_layers
	# python3 scripts/build-components.py
	python3 src/mase_components/activation_layers/test/fixed_gelu_tb.py
	python3 src/mase_components/activation_layers/test/fixed_leaky_relu_tb.py
	python3 src/mase_components/activation_layers/test/fixed_relu_tb.py
	python3 src/mase_components/activation_layers/test/fixed_selu_tb.py
	# DEBUG
	# python3 src/mase_components/activation_layers/test/fixed_sigmoid_tb.py
	# python3 src/mase_components/activation_layers/test/fixed_softermax_1d_tb.py
	# python3 src/mase_components/activation_layers/test/fixed_softermax_tb.py
	# python3 src/mase_components/activation_layers/test/fixed_softmax_tb.py
	python3 src/mase_components/activation_layers/test/fixed_softplus_tb.py
	python3 src/mase_components/activation_layers/test/fixed_softsign_tb.py
	python3 src/mase_components/activation_layers/test/fixed_tanh_tb.py
	# DEBUG softmax based, needs debugging
	# python3 src/mase_components/activation_layers/test/softermax_global_norm_tb.py
	# python3 src/mase_components/activation_layers/test/softermax_local_window_tb.py
	# python3 src/mase_components/activation_layers/test/softermax_lpw_pow2_tb.py
	# python3 src/mase_components/activation_layers/test/softermax_lpw_reciprocal_tb.py
	# python3 src/mase_components/activation_layers/test/test_lint_activation_layers.py
	# python3 src/mase_components/activation_layers/test/test_synth_activation_layers.py
	# activation_layers DEV mode
	# python3 src/mase_components/activation_layers/test/fixed_elu_tb.py
	# python3 src/mase_components/activation_layers/test/fixed_hardshrink_tb.py
	# python3 src/mase_components/activation_layers/test/fixed_hardswish_tb.py
	# python3 src/mase_components/activation_layers/test/fixed_logsigmoid_tb.py
	# python3 src/mase_components/activation_layers/test/fixed_silu_tb.py
	# python3 src/mase_components/activation_layers/test/fixed_softshrink_tb.py
	# fixed-linear layers
	# DEBUG: use bias causes crash
	python3 src/mase_components/fixed_linear_layers/test/fixed_linear_tb.py

reformat:
	# format python files
	black src/chop
	black src/mase_components
	black src/mase_cocotb
	black test
	# format verilog
	# find src/mase_components -name '*.sv' -exec verible-verilog-format --inplace {} +;
