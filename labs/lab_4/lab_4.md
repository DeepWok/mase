<br />
<div align="center">

  <h1 align="center">Lab 4 for Advanced Deep Learning Systems (ADLS) - Software Stream</h1>

  <p align="center">
    ELEC70109/EE9-AML3-10/EE9-AO25
    <br />
		Written by
    <a href="https://aaron-zhao123.github.io/">Aaron Zhao, Cheng Zhang, and Pedro Gimenes </a>
  </p>
</div>


## Implementation tasks

### Task 1
We indeed do not find a speedup by compiling the model before timing a pass.


## Question 3. a
❓ **Question**: How does MXINT8 benefit custom hardware if both the activation and weights in a linear layer are quantized to MXINT8?
MXINT8 if used both for weights and activation allows to transition smoothly between the two operations without having to perform dequantization. It saves computing power and also processing memory as the outputs of the weight stay small.

## Question 3. b
❓ **Question**: What is the purpose of the `dont_need_abs` variable and the `bias` variable? Note that unlike IEEE Floating-Point, MXINT has no implicit leading bit for the mantissa.

`dont_need_abs` is applying a mask to grab the 6th bit from the given mantissa, which plays the role of an explicit leading bit. If it is set to 1, then it corresponds to the normal situation in the case of IEEE Float16, where the leading bit is implicit. `out` corresponds to the IEEE Float16 conversion of the MXINT8 number. If then the 6th bit is 0, we need to compensate for the implicite addition of the leading bit when converting to IEEE Float16. 
We threfore need to substract `bias`, which is the difference between the correct value and the value with an extra leading bit added.
By taking `out` and substracting it with a Float16 with same sign, same exponent and mantissa set to 0, we only substract the influence of the leading implicit bit. 

## Question 3. c
❓ **Question (Challenging)**: How does `cta_tiler` partition the data for copy? You may find the documentation of `local_tile` in CUTE helpful ([ref](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md))


`cta_tiler` has to create tiles that fit exactly one block. It takes a shape of size `BLK_M`, `BLK_K`, which defines the shape of a block. Since a block is a group of threads, and a block generaly contains 1024 threads the tensor needs to be cut into pieces of size at most 1024 : (m x k) where m*k <= 1024. 
Therefore, `m` is set as $m = min\{2^p , 2^p <= group\_size\}$, to fit all of the tensors of the same group in one block, and the tensors of different groups in the same block (`BLK_K` groups).

❓ **Question (Challenging)**: How does `layout_sX` partition the threads in a threadblock for computation? You may find the documentation of `local_partition` in CUTE helpful ([ref](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md))

A block has a layout similar to the tile associated to him, with same 2D shape. One dimension indexes the number of the group (among the groups that are present in that block), and the other the number of the tensor inside that group. 
