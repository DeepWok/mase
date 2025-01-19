Machine-Learning System Exploration Tools
==========================================

Mase is a Machine Learning compiler based on PyTorch FX, maintained by researchers at Imperial College London. We provide a set of tools for inference and training optimization of state-of-the-art language and vision models. The following features are supported, among others:

- **Quantization Search**: mixed-precision quantization of any PyTorch model. We support `microscaling <https://arxiv.org/abs/2310.10537>`__ and other numerical formats, at various granularities.

- **Quantization-Aware Training (QAT)**: finetuning quantized models to minimize accuracy loss.

- **Hardware Generation**: automatic generation of high-performance FPGA accelerators for arbitrary Pytorch models, through the Emit Verilog flow.

- **Distributed Deployment**: Automatic parallelization of models across distributed GPU clusters, based on the `Alpa <https://arxiv.org/abs/2201.12023>`__ algorithm.

For more details, refer to the `Tutorials <https://deepwok.github.io/mase/modules/documentation/tutorials.html>`_. If you enjoy using the framework, you can support us by starring the repository on `GitHub <https://github.com/DeepWok/mase>`__!

Efficient AI Optimization 
----------------------------------------------------

MASE provides a set of composable tools for optimizing AI models. The tools are designed to be modular and can be used in a variety of ways to optimize models for different hardware targets. The tools can be used to optimize models for inference, training, or both. The tools can be used to optimize models for a variety of hardware targets, including CPUs, GPUs, and FPGAs. The tools can be used to optimize models for a variety of applications, including computer vision, natural language processing, and speech recognition.



Hardware Generation
----------------------------------------------------

Machine learning accelerators have been used extensively to compute models with high performance and low power. Unfortunately, the development pace of ML models is much faster than the accelerator design cycle, leading to frequent changes in the hardware architecture requirements, rendering many accelerators obsolete. Existing design tools and frameworks can provide quick accelerator prototyping, but only for a limited range of models that fit into a single hardware device. With the emergence of large language models such as GPT-3, there is an increased need for hardware prototyping of large models within a many-accelerator system to ensure the hardware can scale with ever-growing model sizes.

.. image:: ../imgs/mase_overview.png
   :alt: logo
   :align: center

MASE provides an efficient and scalable approach for exploring accelerator systems to compute large ML models by directly mapping onto an efficient streaming accelerator system. Over a set of ML models, MASE can achieve better energy efficiency to GPUs when computing inference for recent transformer models.


Documentation
----------------------------------------------------

For more details, explore the documentation

.. toctree::
   :maxdepth: 1
   :caption: Overview

   modules/documentation/installation
   modules/documentation/quickstart
   modules/documentation/tutorials
   modules/documentation/health
   modules/documentation/specifications

.. toctree::
   :maxdepth: 2
   :caption: Machop API

   modules/machop

.. toctree::
   :maxdepth: 1
   :caption: Mase Components

   modules/hardware/hardware_documentation

.. toctree::
   :maxdepth: 1
   :caption: Advanced Deep Learning Systems

   modules/adls_2024
   modules/adls_2023