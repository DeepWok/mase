vhls=/scratch/jc9016/tools/Xilinx/2020.2
user=$(if $(shell id -u),$(shell id -u),9001)
group=$(if $(shell id -g),$(shell id -g),1000)

# Make sure the repo is up to date
sync:
	git submodule sync
	git submodule update --init --recursive

# Build Docker container
build-docker: 
	(cd Docker; docker build --build-arg UID=$(user) --build-arg GID=$(group) . --tag jc-centos8)

shell: build-docker
	docker run -it -u $(user) -v $(shell pwd):/workspace jc-centos8:latest /bin/bash

build:
	bash scripts/build-llvm.sh

clean:
	rm -rf llvm/build
