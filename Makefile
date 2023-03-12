vhls=/scratch/jc9016/tools/Xilinx/2020.2
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
	(cd Docker; docker build --build-arg UID=$(user) --build-arg GID=$(group) . --tag mase-ubuntu2204)

shell: build-docker
	docker run -it --hostname mase-ubuntu2204 -u $(user) -v $(shell pwd):/workspace mase-ubuntu2204:latest /bin/bash 

build:
	bash scripts/build-llvm.sh
	bash scripts/build-mase-hls.sh
	bash scripts/build-air-aie.sh

clean:
	rm -rf llvm/build
