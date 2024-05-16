# Group Norm 2D

The `group_norm_2d` module implements the [GroupNorm PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html#groupnorm) layer with the assumption that `affine=False`.

With this assumption, this module is able to be specialized into [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) and [InstanceNorm](https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html) modules simply by setting the number of `GROUP_CHANNELS` to the channel dimension `C` or `1` respectively.

Note that the `group_norm_2d` component is an internal hardware component and is exposed to the rest of the MASE software stack through the unified `norm` component.

## Overview

The `group_norm_2d` is a fully pipelined module which follows the dataflow streaming protocol.

<p align="center">
  <img src="https://raw.githubusercontent.com/DeepWok/mase/main/docs/source/imgs/hardware/group_norm_2d.png" alt="img">
</p>
