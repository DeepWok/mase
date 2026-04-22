NUM_WORKERS ?= 1

sw_test_dir = test/

test-sw:
	pytest --log-level=DEBUG --verbose \
		-n $(NUM_WORKERS) \
		-m "not large" \
		--cov=src/chop/ --cov-report=html \
		--html=report.html --self-contained-html \
		--junitxml=software_report.xml \
		--profile --profile-svg \
		$(sw_test_dir)

clean:
	rm -rf tmp mase_output
