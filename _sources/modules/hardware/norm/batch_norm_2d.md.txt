# Batch Norm 2D

The `batch_norm_2d` module implements the [BatchNorm2d PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) layer with the assumption that `affine=False`.

Note that the `batch_norm_2d` component is an internal hardware component and is exposed to the rest of the MASE software stack through the unified `norm` component.

## Overview

The `batch_norm_2d` is a fully pipelined module which follows the dataflow streaming protocol.

<p align="center">
  <img src="https://raw.githubusercontent.com/DeepWok/mase/main/docs/source/imgs/hardware/batch_norm_2d.png" alt="img">
</p>
