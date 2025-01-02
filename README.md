# Machine-Learning Accelerator System Exploration Tools

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Doc][doc-shield]][doc-url]

[contributors-shield]: https://img.shields.io/github/contributors/DeepWok/mase.svg?style=flat
[contributors-url]: https://github.com/DeepWok/mase/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/DeepWok/mase.svg?style=flat
[forks-url]: https://github.com/DeepWok/mase/network/members
[stars-shield]: https://img.shields.io/github/stars/DeepWok/mase.svg?style=flat
[stars-url]: https://github.com/DeepWok/mase/stargazers
[issues-shield]: https://img.shields.io/github/issues/DeepWok/mase.svg?style=flat
[issues-url]: https://github.com/DeepWok/mase/issues
[license-shield]: https://img.shields.io/github/license/DeepWok/mase.svg?style=flat
[license-url]: https://github.com/DeepWok/mase/blob/master/LICENSE.txt
[issues-shield]: https://img.shields.io/github/issues/DeepWok/mase.svg?style=flat
[issues-url]: https://github.com/DeepWok/mase/issues
[doc-shield]: https://readthedocs.org/projects/pytorch-geometric/badge/?version=latest
[doc-url]: https://deepwok.github.io/mase/

## Overview

Mase is a Machine Learning compiler based on PyTorch FX, maintained by researchers at Imperial College London. We provide a set of tools for inference and training optimization of state-of-the-art language and vision models. The following features are supported, among others:

- Efficient AI Optimization: 
  MASE provides a set of composable tools for optimizing AI models. The tools are designed to be modular and can be used in a variety of ways to optimize models for different hardware targets. The tools can be used to optimize models for inference, training, or both. We support features such as the following:

  - Quantization Search: mixed-precision quantization of any PyTorch model. We support microscaling and other numerical formats, at various granularities.
  - Quantization-Aware Training (QAT): finetuning quantized models to minimize accuracy loss.
  - And more!

- Hardware Generation: automatic generation of high-performance FPGA accelerators for arbitrary Pytorch models, through the Emit Verilog flow.

- Distributed Deployment (Beta): Automatic parallelization of models across distributed GPU clusters, based on the Alpa algorithm.

For more details, refer to the Tutorials. If you enjoy using the framework, you can support us by starring the repository on GitHub!


## MASE Publications

* Fast Prototyping Next-Generation Accelerators for New ML Models using MASE: ML Accelerator System Exploration, [link](https://arxiv.org/abs/2307.15517)
  ```
  @article{cheng2023fast,
  title={Fast prototyping next-generation accelerators for new ml models using mase: Ml accelerator system exploration},
  author={Cheng, Jianyi and Zhang, Cheng and Yu, Zhewen and Montgomerie-Corcoran, Alex and Xiao, Can and Bouganis, Christos-Savvas and Zhao, Yiren},
  journal={arXiv preprint arXiv:2307.15517},
  year={2023}}
  ```
* MASE: An Efficient Representation for Software-Defined ML Hardware System Exploration, [link](https://openreview.net/forum?id=Z7v6mxNVdU)
  ```
  @article{zhangmase,
  title={MASE: An Efficient Representation for Software-Defined ML Hardware System Exploration},
  author={Zhang, Cheng and Cheng, Jianyi and Yu, Zhewen and Zhao, Yiren}}
  ```
### Repository structure

This repo contains the following directories:
* `src/chop` - MASE's software stack
* `src/mase_components` - Internal hardware library
* `src/mase_cocotb` - Internal hardware testing flow
* `src/mase_hls` - HLS component of MASE
* `scripts` - Run and test scripts  
* `test` - Unit testing 
* `docs` - Documentation
* `mlir-air` - MLIR AIR for ACAP devices
* `setup.py` - Installation entry point
* `Docker` - Docker container configurations

## MASE Dev Meetings

* Direct [Google Meet link](meet.google.com/fke-zvii-tgv)
* Join the [Mase Slack](https://join.slack.com/t/mase-tools/shared_invite/zt-2gl60pvur-pktLLLAsYEJTxvYFgffCog)
* If you want to discuss anything in future meetings, please add them as comments in the [meeting agenda](https://docs.google.com/document/d/12m96h7gOhhmikniXIu44FJ0sZ2mSxg9SqyX-Uu3s-tc/edit?usp=sharing) so we can review and add them.

## Donation  

If you think MASE is helpful, please [donate](https://www.buymeacoffee.com/mase_tools) for our work, we appreciate your support!

<img src='./docs/imgs/bmc_qr.png' width='250'>
