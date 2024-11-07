## Overview

This directory contains code designed for transformations in Spiking Neural Networks (SNNs) within the MASE framework. The code adapts and integrates modules from the SpikingJelly library, providing essential functionality for efficient SNN simulations and operations.

## Directory Structure

### `\functional`

- **Description**: This folder contains adapted code from SpikingJelly, offering a functional style interface.
- **Key Features**:
  - Contains containers that allow users to set simulation behavior, either as a **single-step** or **multi-step** process.
  - Provides functional style implementations for SNN layers, enabling modular and flexible usage in various network architectures.

### `\module`

- **Description**: This folder includes module-based code adapted from SpikingJelly.
- **Key Features**:
  - Contains modular containers similar to those in the `functional` directory, allowing the specification of single-step or multi-step simulations.
  - Provides neural network modules specifically designed for SNNs, allowing users to build structured SNN models with ease.
  - Provides neural scaling modules for ann to snn conversion.

### `\auto_cuda`

- **Description**: This folder is directly sourced from SpikingJelly without modifications.
- **Key Features**:
  - Defines acceleration kernels to optimize SNN operations, particularly useful for using CUDA capabilities, cupy or just pytorch.


### `neuron.py`

- **Description**: This file is directly sourced from SpikingJelly without modifications.
- **Key Features**:
  - Defines neuron modules.

### `base.py`

- **Description**: This file is directly sourced from SpikingJelly without modifications.
- **Key Features**:
  - Defines the base class for neuron modules (modules that has internal memory across inference) and the base class for modular-style containers.

### `configuration.py`

- **Description**: This file is directly sourced from SpikingJelly without modifications.
- **Key Features**:
  - Defines various configuration variables used in SpikingJelly, such as settings for the number of CUDA threads.

### `cuda_utils.py`

- **Description**: This file is directly sourced from SpikingJelly without modifications.
- **Key Features**:
  - Contains helper functions for running SNN modules with CUDA.


## Additional Information

This codebase is built on top of SpikingJelly