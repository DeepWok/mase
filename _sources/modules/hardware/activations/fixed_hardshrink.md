# Fixed-Point Hardshrink Layer

The `fixed_hardshrink` module implements the [Pytorch Hardshrink](https://pytorch.org/docs/stable/generated/torch.nn.Hardshrink.html) layer. Given the input vector $x \in \R^{m x n}$, `fixed_hardshrink` returns $y \in \R^{m x n}$. 

$\text{HardShrink}(x) =
\begin{cases}
x, \text{ if } x \gt \lambda \\
x, \text{ if } x \lt -\lambda \\
0, \text{ otherwise }
\end{cases}
$

This function is applied elementwise to the input tensor. For example, taking $m = n = 2$:

Hardshrink($x$) = $ Hardshrink(\begin{bmatrix}
x_1 & x_2 \\
x_3 & x_4 \\
\end{bmatrix})$ = $ \begin{bmatrix} Hardshrink(x_1) & Hardshrink(x_2) \\
Hardshrink(x_3) & Hardshrink(x_4) \\
\end{bmatrix}$


## Overview

The `fixed_hardshrink` module follows the dataflow streaming protocol and works by evaluating the condition and then computing a single fixed point addition to compute the Hardshrink function in a single cycle with input and output quantization to the desired precision, it employs a casting module (fixed_rounding.sv) to implement the desired output precision.

The module has the following parameters, following the hardware metadata standard (see [here](https://deepwok.github.io/mase/modules/chop/analysis/add_metadata.html#add-hardware-metadata-analysis-pass)). Besides `PRECISION_DIM_*` parameters, which dictate the numerical precision, and `TENSOR_SIZE_DIM_*`, which is directly inferred from Pytorch tensor shapes, the following parameters can be adjusted to affect hardware performance and functionality.

| Parameter                    	| Default Value            	| Definition                                                                                                                                                                                                                                     	|
|------------------------------	|--------------------------	|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| LAMBDA  	| 0.5                        	| A parameter in the Pytorch module, controls the threshold of shrinkage.                                    | DATA_IN_0_PARALLELISM_DIM_0  	| 4                        	| Number of elements per transaction at the input interface. Impacts the area usage by increasing the required FIFO length (only required with different input and output parallelisms)                                                                   |
| DATA_IN_0_PARALLELISM_DIM_1  	| 4                        	| Number of elements per transaction at the input interface. Impacts the area usage by increasing the required FIFO length (only required with different input and output parallelisms)                                                                      |
| DATA_IN_0_PARALLELISM_DIM_0  	| DATA_IN_0_PARALLELISM_DIM_0                        	| Number of elements per transaction at the output interface, this is what controls the number of read-only memories or LUTs that are instantiated.                                                                    |
| DATA_OUT_0_PARALLELISM_DIM_1       	| DATA_IN_0_PARALLELISM_DIM_1 	| Number of elements per transaction at the output interface, this is what controls the number of adders and arithmetic necessary for checking the condition.                                                                                                                                     

## <a name="latency_analaysis"></a> Latency Analysis

The time taken to compute a Hardshrink layer using the `fixed_hardshrink` module, $L_{Hardshrink}$ depends entirely on the operating frequency of the hardware. The implemented designs operate combinatorially, for very high bitwidths and a high FMax, the module may need to be pipelined, however, this can be done trivially. The latency can be calculated as follows: $\frac{1}{f_{clk}}$ where $f_{clk}$ is the operating frequency of the hardware.