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
	# dev mode
	# python3 src/mase_components/activations/test/fixed_elu_tb.py
	python3 src/mase_components/activations/test/fixed_gelu_tb.py
	# dev mode
	# python3 src/mase_components/activations/test/fixed_hardshrink_tb.py
	# dev mode
	# python3 src/mase_components/activations/test/fixed_hardswish_tb.py
	python3 src/mase_components/activations/test/fixed_leaky_relu_tb.py
	# dev mode
	# python3 src/mase_components/activations/test/fixed_logsigmoid_tb.py
	python3 src/mase_components/activations/test/fixed_relu_tb.py
	python3 src/mase_components/activations/test/fixed_selu_tb.py
	# DEBUG needs debugging
	# python3 src/mase_components/activations/test/fixed_sigmoid_tb.py
	# dev mode
	# python3 src/mase_components/activations/test/fixed_silu_tb.py
	# DEBUG softmax based, needs debugging
	# python3 src/mase_components/activations/test/fixed_softermax_1d_tb.py
	# python3 src/mase_components/activations/test/fixed_softermax_tb.py
	# python3 src/mase_components/activations/test/fixed_softmax_tb.py
	python3 src/mase_components/activations/test/fixed_softplus_tb.py
	# dev mode
	# python3 src/mase_components/activations/test/fixed_softshrink_tb.py
	python3 src/mase_components/activations/test/fixed_softsign_tb.py
	python3 src/mase_components/activations/test/fixed_tanh_tb.py
	# DEBUG softmax based, needs debugging
	# python3 src/mase_components/activations/test/softermax_global_norm_tb.py
	# python3 src/mase_components/activations/test/softermax_local_window_tb.py
	# python3 src/mase_components/activations/test/softermax_lpw_pow2_tb.py
	# python3 src/mase_components/activations/test/softermax_lpw_reciprocal_tb.py
	# python3 src/mase_components/activations/test/test_lint_activations.py
	# python3 src/mase_components/activations/test/test_synth_activations.py

reformat:
	# format python files
	black src/chop
	black src/mase_components
	# format verilog
	# find src/mase_components -name '*.sv' -exec verible-verilog-format --inplace {} +;