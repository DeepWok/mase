# Normalization Module

The `norm` module is a wrapper which acts as the single interface for all normalization hardware modules. It current supports the instantiation of these normalization layers:

- [BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
- [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
- [GroupNorm](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html#groupnorm)
- [InstanceNorm](https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html)
- [RMSNorm](https://arxiv.org/abs/1910.07467)

Note that it implements all normalization layers with the assumption that `affine=false`.

## Parameter Overview

The module has the following parameters, following the hardware metadata standard (see [here](https://deepwok.github.io/mase/modules/api/analysis/add_metadata.html#add-hardware-metadata-analysis-pass)). Besides `PRECISION_DIM_*` parameters, which dictate the numerical precision, and `TENSOR_SIZE_DIM_*`, which is directly inferred from Pytorch tensor shapes, the following parameters can be adjusted to affect hardware performance.

| Parameter                   | Description |
| -----                       | -----       |
| NORM_TYPE                   | Selects the type of normalization layer. Set to `"BATCH_NORM"`, `"LAYER_NORM"`, `"INSTANCE_NORM"`, `"GROUP_NORM"`, or `"RMS_NORM"`. |
| ISQRT_LUT_MEMFILE           | Sets to path to the initialization LUT for inverse square root unit. |
| SCALE_LUT_MEMFILE           | (Only used in `BATCH_NORM`) Sets scale LUT mem file. |
| SHIFT_LUT_MEMFILE           | (Only used in `BATCH_NORM`) Sets shift LUT mem file. |
| DATA_IN_0_PARALLELISM_DIM_0 | Hardware compute parallelism across 1st spatial dimension. |
| DATA_IN_0_PARALLELISM_DIM_1 | Hardware compute parallelism across 2nd spatial dimension. |
| DATA_IN_0_PARALLELISM_DIM_2 | Number of channels to normalize across. For `LAYER_NORM`, set to the number of total channels. For `GROUP_NORM`, set to total channels divided by number of groups. |
| DATA_IN_0_PRECISION_0       | Input data width. |
| DATA_IN_0_PRECISION_1       | Input fractional data width, must be <= input data width. |
| DATA_OUT_0_PRECISION_0      | Output data width. |
| DATA_OUT_0_PRECISION_1      | Output fractional data width, must be <= output data width. |
