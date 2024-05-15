
# Fixed-Point SiLU Layer

The `fixed_silu` module implements the [Pytorch SiLU](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html) layer. Given the input vector $x \in \R^{m x n}$, `fixed_silu` returns $y \in \R^{m x n}$. 

SiLU(x) $\coloneqq$ x * $\sigma$(x) 
  
where $\sigma$ is the sigmoid function defined as $\sigma(x) = \frac{1}{1 + e^{-x}}$

This function is applied elementwise to the input tensor. For example, taking $m = n = 2$:

SiLU($x$) = $ SiLU(\begin{bmatrix}
x_1 & x_2 \\
x_3 & x_4 \\
\end{bmatrix})$ = $ \begin{bmatrix} SiLU(x_1) & SiLU(x_2) \\
SiLU(x_3) & SiLU(x_4) \\
\end{bmatrix}$


## Overview

The `fixed_silu` module follows the dataflow streaming protocol and works by employing a table mapping which is used to compute the SiLU function. The table is generated during the emit verilog pass and is used to compute the SiLU function in a single cycle with input and output quantization to the desired precision. This can be instantiated using LUTs or BRAM (memory).


The module has the following parameters, following the hardware metadata standard (see [here](https://deepwok.github.io/mase/modules/api/analysis/add_metadata.html#add-hardware-metadata-analysis-pass)). Besides `PRECISION_DIM_*` parameters, which dictate the numerical precision, and `TENSOR_SIZE_DIM_*`, which is directly inferred from Pytorch tensor shapes, the following parameters can be adjusted to affect hardware performance.

| Parameter                    	| Default Value            	| Definition                                                                                                                                                                                                                                     	|
|------------------------------	|--------------------------	|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| DATA_IN_0_PARALLELISM_DIM_0  	| 4                        	| Number of elements per transaction at the input interface. Impacts the area usage by increasing the required FIFO length (only required with different input and output parallelisms)                                                                   |
| DATA_IN_0_PARALLELISM_DIM_1  	| 4                        	| Number of elements per transaction at the input interface. Impacts the area usage by increasing the required FIFO length (only required with different input and output parallelisms)                                                                      |
| DATA_IN_0_PARALLELISM_DIM_0  	| DATA_IN_0_PARALLELISM_DIM_0                        	| Number of elements per transaction at the output interface, this is what controls the number of read-only memories or LUTs that are instantiated.                                                                    |
| DATA_OUT_0_PARALLELISM_DIM_1       	| DATA_IN_0_PARALLELISM_DIM_1 	| Number of elements per transaction at the output interface, this is what controls the number of read-only memories or LUTs that are instantiated.                                                                                                                                     

## <a name="latency_analaysis"></a> Latency Analysis

The time taken to compute a SiLU layer using the `fixed_silu` module, $L_{SiLU}$ depends entirely on the operating frequency of the hardware. The implemented designs operate combinatorially, this will be ideal in the majority of cases due to the read only nature of the memories and lack of arithemtic logic resulting in a short critical path. The latency can be calculated as follows: $\frac{1}{f_{clk}}$ where $f_{clk}$ is the operating frequency of the hardware.