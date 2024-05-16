# RMS Norm 2D

The `rms_norm_2d` module implements [RMSNorm](https://arxiv.org/abs/1910.07467).

Note that the `rms_norm_2d` component is an internal hardware component and is exposed to the rest of the MASE software stack through the unified `norm` component.

## Overview

The `rms_norm_2d` is a fully pipelined module which follows the dataflow streaming protocol.

<p align="center">
  <img src="https://raw.githubusercontent.com/DeepWok/mase/main/docs/source/imgs/hardware/rms_norm_2d.png" alt="img">
</p>
