alias ts := test-sw
alias re := reformat

test-sw:
	# cmd line interface is no longer supported
	# bash scripts/test-machop.sh
	pytest --log-level=DEBUG --verbose \
		-n 1 \
		--cov=src/chop/ --cov-report=html \
		--html=report.html --self-contained-html \
		--junitxml=test/report.xml \
		--profile --profile-svg \
		test/

reformat:
	# format python files
	black *.py
	black src/chop
	black src/mase_components
	black src/mase_cocotb
	black test
	# format verilog
	# find src/mase_components -name '*.sv' -exec verible-verilog-format --inplace {} +;
