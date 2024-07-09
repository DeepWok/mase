alias ts := test-sw
alias th := test-hw

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
	python src/mase_components/activations/test/fixed_gelu_tb.py
	python src/mase_components/activations/test/fixed_relu_tb.py
	# dev mode
	# python src/mase_components/activations/test/fixed_hardshrink_tb.py
	# dev mode
	# python src/mase_components/activations/test/fixed_hardswish_tb.py
	python src/mase_components/activations/test/fixed_leaky_relu_tb.py