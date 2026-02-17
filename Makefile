local ?= 0

GPU_AVAILABLE := $(shell command -v nvidia-smi 2> /dev/null)

# * Check if a GPU is available
ifeq ($(GPU_AVAILABLE),)
    PLATFORM := cpu
else
    PLATFORM := gpu
endif

# * Set docker image according to local flag
# * If local is set, should run locally built image
# * Otherwise pull from dockerhub
ifeq ($(local), 1)
    img = "mase-ubuntu2204:latest"
else
    img = "deepwok/mase-docker-$(PLATFORM):latest"
endif

# * Check if running on a mac to set user path
ifeq ($(shell uname),Darwin)
    USER_PREFIX=Users
else
    USER_PREFIX=home
endif

coverage=test/

sw_test_dir = test/

NUM_WORKERS ?= 1

# Make sure the repo is up to date
sync:
	git submodule sync
	git submodule update --init --recursive

# Build Docker container
build-docker:
	if [ $(local) -eq 1 ]; then \
		if [ ! -d Docker ]; then \
    			git clone git@github.com:jianyicheng/mase-docker.git Docker; \
		fi; \
		docker build -f Docker/Dockerfile-$(PLATFORM) --tag mase-ubuntu2204 Docker; \
	else \
		docker pull $(img); \
	fi

shell:
	docker run -it --shm-size 256m \
        --hostname mase-ubuntu2204 \
        -w /workspace \
        -v /$(USER_PREFIX)/$(shell whoami)/.gitconfig:/root/.gitconfig \
        -v /$(USER_PREFIX)/$(shell whoami)/.ssh:/root/.ssh \
        -v /$(USER_PREFIX)/$(shell whoami)/.mase:/root/.mase:z \
        $(img) /bin/bash

test-sw:
	pytest --log-level=DEBUG --verbose \
		-n $(NUM_WORKERS) \
		-m "not large" \
		--cov=src/chop/ --cov-report=html \
		--html=report.html --self-contained-html \
		--junitxml=software_report.xml \
		--profile --profile-svg \
		$(sw_test_dir)

test-all: test-sw
	mkdir -p ./tmp
	(cd tmp; python3 ../scripts/test-torch-mlir.py || exit 1)

build:
	bash scripts/build-llvm.sh || exit 1

clean:
	rm -rf llvm
	rm -rf tmp mase_output
