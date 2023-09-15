vhls=/mnt/applications/Xilinx/23.1
user=$(if $(shell id -u),$(shell id -u),9001)
group=$(if $(shell id -g),$(shell id -g),1000)

# Make sure the repo is up to date
sync:
	git submodule sync
	git submodule update --init --recursive
	bash mlir-air/utils/clone-llvm.sh 
	bash mlir-air/utils/clone-mlir-aie.sh 

# Build Docker container
build-docker: 
	docker build --build-arg UID=$(user) --build-arg GID=$(group) --build-arg VHLS_PATH=$(vhls) -f Docker/Dockerfile --tag mase-ubuntu2204 Docker

shell: build-docker
	docker run -it --shm-size 256m --hostname mase-ubuntu2204 -u $(user) -v $(vhls):$(vhls) -v $(shell pwd):/workspace mase-ubuntu2204:latest /bin/bash 

shell-kraken: 
	docker build --build-arg UID=$(user) --build-arg GID=$(group) --build-arg VHLS_PATH=$(vhls) -f Docker/Dockerfile-kraken --tag mase-ubuntu2204 Docker
	docker run -it --shm-size 256m --hostname mase-ubuntu2204 -w /workspace -v $(shell pwd):/workspace:z mase-ubuntu2204:latest /bin/bash 

build-vagrant: 
	(cd vagrant; vagrant up)

shell-vagrant: 
	(cd vagrant; vagrant ssh)

test-hw:
	python3 scripts/test-hardware.py -a || exit 1

test-sw:
	bash scripts/test-machop.sh || exit 1

test-all: test-hw test-sw
	python3 scripts/test-torch-mlir.py || exit 1

build:
	bash scripts/build-llvm.sh
	bash scripts/build-mase-hls.sh
	bash scripts/build-aie.sh
	bash scripts/build-air.sh

clean:
	rm -rf llvm/build
	rm -rf mlir-air/build
	rm -rf mlir-aie/build
	rm -rf hls/build
