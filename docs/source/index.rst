Machine-Learning System Exploration Tools
==========================================

Mase is a machine learning optimization framework based on PyTorch FX, maintained by researchers at Imperial College London. It provides a modular set of tools for training and inference optimization of state-of-the-art language and vision models.

The following capabilities are supported:

- **MX Post-Training Quantization (PTQ)**: quantize models to MX formats (MXINT, MXFP) with optional GPTQ weight calibration and rotation-based activation outlier mitigation.

- **Quantization-Aware Training (QAT)**: finetune quantized models to recover accuracy after quantization.

- **Mixed-Precision Search**: automatically search for the best per-layer precision assignment using NAS-style search with Optuna.

- **LoRA Fine-tuning**: parameter-efficient finetuning of large language models.

- **Pruning**: structured and unstructured pruning of model weights.

For a hands-on introduction, refer to the `Tutorials <https://deepwok.github.io/mase/modules/documentation/tutorials.html>`_. If you enjoy using the framework, please star the repository on `GitHub <https://github.com/DeepWok/mase>`__!


Documentation
----------------------------------------------------

For more details, explore the documentation

.. toctree::
   :maxdepth: 1
   :caption: Overview

   modules/documentation/installation
   modules/documentation/quickstart
   modules/documentation/tutorials
   modules/documentation/specifications


.. toctree::
   :maxdepth: 1
   :caption: Advanced Deep Learning Systems

   modules/adls_2024
   modules/adls_2023
