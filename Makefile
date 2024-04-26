vhls=/mnt/applications/Xilinx/23.1
vhls_version=2023.1
local=0
target=cpu
img=$(if $local,"mase-ubuntu2204:latest","deepwok/mase-docker-$(target):latest")
user=$(if $(shell id -u),$(shell id -u),9001)
group=$(if $(shell id -g),$(shell id -g),1000)

sw_test_dir = machop/test/
hw_test_dir = machop/mase_components/

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
	if [ $(local) = 1 ]; then \
		if [ ! -d Docker ]; then \
    			git clone git@github.com:jianyicheng/mase-docker.git Docker; \
		fi; \
		docker build --build-arg VHLS_PATH=$(vhls) --build-arg VHLS_VERSION=$(vhls_version) -f Docker/Dockerfile-$(target) --tag mase-ubuntu2204 Docker; \
	else \
		docker pull docker.io/deepwok/mase-docker-$(target):latest; \
	fi

shell: build-docker
	docker run -it --shm-size 256m \
        --hostname mase-ubuntu2204 \
        -w /workspace \
        -v $(vhls):$(vhls) \
        -v /home/$(shell whoami)/.gitconfig:/root/.gitconfig \
        -v /home/$(shell whoami)/.ssh:/root/.ssh \
        -v $(shell pwd):/workspace:z \
        $(img) /bin/bash

test-hw:
	pytest --log-level=DEBUG --verbose \
		-n 1 \
		--html=report.html --self-contained-html \
		$(hw_test_dir)

test-sw:
	bash scripts/test-machop.sh
	pytest --log-level=DEBUG --verbose \
		-n 1 \
		--cov=machop/chop/ --cov-report=html \
		--html=report.html --self-contained-html \
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
