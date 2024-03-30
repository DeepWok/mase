# quantized_matmul.sv
This module constructs the low-precision processing unit in [`llm_int8_top`](../rtl/llm_int8_top.sv). It performs the following operaitons:

**(1) Quantization:** this FP16 activations $X_{f16}$ and weight matrix $W_{f16}$ as input, and quantize them to matrices $X_{i8}$ and $W_{i8}$ of Int8 precision:

$$ X_{i8} = \left\lfloor \frac{127 \cdot X_{f16}}{c_x}\right\rceil$$

$$ W_{i8} = \left\lfloor \frac{127 \cdot W_{f16}}{c_w}\right\rceil$$

where 
$$ c_x = \max\left(\left|X_{f16_{i, j}}\right|\right) $$
$$ c_w = \max\left(\left|W_{f16_{i, j}}\right|\right) $$

and  $\left\lfloor\right\rceil$ indicates rounding to the nearest integer.

**(2) Matmul & Dequantization:** This performs Int8 matrix multiplication, and de-quantize the output matrix back to FP16 precision:

$$ X_{i8}W_{i8} = Out_{i32}$$
$$ \frac{Out_{i32}*(c_x*c_w)}{127*127} = Out_{f16} $$



## Structure Overview
![](./figs/top_level.png)


The component consists of two sub-modules: 
* `quantizer_top`, which searches the maximum absolute number $c_x$ of the input matrix.



* `fixed_matmul_core_dequant`

These two submodules all both pipelined and communite with each other via hand-shake protocol.

### 1. `find_max`

### 2. `quantizer_part`
This is a fully combinational circuit which multiplies all elements of `data_in` in parallel with the common scaling constant $127/c_x$. The `fixed_rounding` component rounds all scaled data to Int8 precision.

Theoretical latency of this component is 1 cycle.


## Module Specifications
### Ports
1. Input Ports:
    * `clk` & `rst`
    * `data_in`: input vector (or equivalently a flattened matrix) with default precision FP16.
    * `data_in_valid` & `data_in_ready`: handshake signals.
2. Output Ports:
    * `data_out`: quantized vector (or equivalently a flattened matrix) with default precision Int8.
    * `data_out_valid` & `data_out_ready`: handshake signals.

### Parameters

| Parameter | Default Value | Definition |
| :---: | :---: | :---:|
| IN_WIDTH | 16 | Data width of input matrix $X_{f16}$|
| IN_SIZE | 4 | Column size of input (output) matrix |
| IN_PARALLELISM | 1 | Row size of input (output) matrix |
| MAX_NUM_WIDTH | IN_WIDTH | Data width of the max number $c_x$|
| QUANTIZATION_WIDTH | 8 | Data width of output quantized matrix $X_{i8}$|

