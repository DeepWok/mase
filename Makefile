vhls=/mnt/applications/Xilinx/23.1
vhls_version=2023.1
local=0
img=$(if $local,"mase-ubuntu2204:latest","deepwok/mase-docker:latest")
user=$(if $(shell id -u),$(shell id -u),9001)
group=$(if $(shell id -g),$(shell id -g),1000)
coverage=machop/test/

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
	if [ $(local) ]; then \
		docker build --build-arg VHLS_PATH=$(vhls) --build-arg VHLS_VERSION=$(vhls_version) -f Docker/Dockerfile --tag mase-ubuntu2204 Docker; \
	else \
		docker pull docker.io/deepwok/mase-docker:latest; \
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

# There is a historical reason that test files are stored under the current directory
# Short-term solution: call scripts under /tmp so we can clean it properly
test-hw:
	mkdir -p ./tmp
	pip install .
	(cd tmp; python3 ../scripts/test-hardware.py -a || exit 1)

test-sw:
	pytest --log-level=DEBUG --verbose -n 1 --cov=machop/chop/ --cov-report=html $(coverage) --html=report.html --self-contained-html --profile --profile-svg

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
