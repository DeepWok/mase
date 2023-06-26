# Machine-Learning Accelerator System Exploration Tools

This repo contains the following directories:
* `examples` - NN models by PyTorch
* `hardware` - Internal hardware library 
* `software` - Torch based training library 
* `scripts` - Installation scripts  
* `hls` - HLS part of the Mase  
* `mlir-air` - MLIR air for ACAP devices  
* `docs` - Documentation  
* `Docker` - Docker container configurations  
* `vagrant` - Vagrant box configurations  

## Overview
![Alt text](./docs/imgs/overview.png)

## Getting Started

First, make sure the repo is up to date:
```sh
make sync
```
Start with the docker container by running the following command under the repo:
```sh
make shell
```
It may take long time to build the docker container for the first time. Once done, you should enter the docker container. To build the tool, run the following command:
```sh
cd /workspace
make build
```
This should also take long time to finish.

If you would like to contribute, please check the [wiki](https://github.com/JianyiCheng/mase-tools/wiki) for more information.

### Toy example of running a neural network

TODO
