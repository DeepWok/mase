vhls=/scratch/jc9016/tools/Xilinx/2020.2
user=$(if $(shell id -u),$(shell id -u),9001)
group=$(if $(shell id -g),$(shell id -g),1000)

# Build Docker container
build-docker:
	(cd Docker; docker build --build-arg UID=$(user) --build-arg GID=$(group) . --tag jc-centos8)

shell: build-docker
	docker run -it -u $(user) -v $(shell pwd):/workspace jc-centos8:latest /bin/bash

