vhls=/mnt/applications/Xilinx/23.1
vhls_version=2023.1
local ?= 0

GPU_AVAILABLE := $(shell command -v nvidia-smi 2> /dev/null)
VIVADO_AVAILABLE := $(shell command -v vivado 2> /dev/null)

# * Check if a GPU is available
ifeq ($(GPU_AVAILABLE),)
    PLATFORM := cpu
else
    PLATFORM := gpu
endif

# * Mount Vivado HLS path only if Vivado is available (to avoid path not found errors)
# Include shared folder containing board files etc
ifeq ($(VIVADO_AVAILABLE),)
    DOCKER_RUN_EXTRA_ARGS=
else
    DOCKER_RUN_EXTRA_ARGS= -v /mnt/applications/:/mnt/applications -v $(vhls):$(vhls)
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
hw_test_dir = src/mase_components/

NUM_WORKERS ?= 1

# Make sure the repo is up to date
sync:
	git submodule sync
	git submodule update --init --recursive

# Only needed if you are using the MLIR flow - it will be slow!
sync-mlir:
	bash mlir-air/utils/github-clone-build-libxaie.sh
	bash mlir-air/utils/clone-llvm.sh 
	bash mlir-air/utils/clone-mlir-aie.sh 

# Build Docker container
build-docker:
	if [ $(local) -eq 1 ]; then \
		if [ ! -d Docker ]; then \
    			git clone git@github.com:jianyicheng/mase-docker.git Docker; \
		fi; \
		docker build --build-arg VHLS_PATH=$(vhls) --build-arg VHLS_VERSION=$(vhls_version) -f Docker/Dockerfile-$(PLATFORM) --tag mase-ubuntu2204 Docker; \
	else \
		docker pull $(img); \
	fi

shell:
	docker run -it --shm-size 256m \
        --hostname mase-ubuntu2204 \
        -w /workspace \
        -v /$(USER_PREFIX)/$(shell whoami)/.gitconfig:/root/.gitconfig \
        -v /$(USER_PREFIX)/$(shell whoami)/.ssh:/root/.ssh \
        -v /$(USER_PREFIX)/$(shell whoami)/.mase:/root/.mase \
        -v $(shell pwd):/workspace:z \
        $(DOCKER_RUN_EXTRA_ARGS) \
        $(img) /bin/bash

test-hw:
	pytest --log-level=DEBUG --verbose \
		-n $(NUM_WORKERS) \
		--junitxml=hardware_report.xml \
		--html=report.html --self-contained-html \
		$(hw_test_dir)

test-sw:
	bash scripts/test-machop.sh
	pytest --log-level=DEBUG --verbose \
		-n $(NUM_WORKERS) \
		--cov=src/chop/ --cov-report=html \
		--html=report.html --self-contained-html \
		--junitxml=software_report.xml \
		--profile --profile-svg \
		$(sw_test_dir)

test-all: test-hw test-sw
	mkdir -p ./tmp
	(cd tmp; python3 ../scripts/test-torch-mlir.py || exit 1)

build:
	bash scripts/build-llvm.sh || exit 1
	bash scripts/build-mase-hls.sh || exit 1

build-aie:
	bash scripts/build-aie.sh || exit 1
	bash scripts/build-air.sh || exit 1

clean:
	rm -rf llvm
	rm -rf aienginev2 mlir-air/build mlir-aie
	rm -rf hls/build
	rm -rf vck190_air_sysroot
	rm -rf tmp mase_output
