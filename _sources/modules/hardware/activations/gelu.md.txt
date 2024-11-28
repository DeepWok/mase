# [GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)

The GELU function is defined as follows:

`GELU(x) = x * Φ(x)`

where:
- Φ(x) is the Cumulative Distribution Function for the Gaussian Distribution.

When the approximate argument is set to 'tanh', GELU is estimated with:

`GELU(x) = 0.5 * x * (1 + Tanh(2/π * (x + 0.044715 * x^3)))`

### Parameters:

- `approximate` (str, optional): The GELU approximation algorithm to use: 'none' | 'tanh'. Default: 'none'.

## Implementation

Similarly to Softsign, due to the complexity of the computation, piecewise quadratic approximation was used. Analysis of function plot carries the following observations. For all negative values up to approximately -4, the function takes the value of 0. Furthermore, for values larger than 4, the output is equal to the input.

Those properties allow us to only do piecewise approximations for the range of (-4,4). The function was approximated in intervals of 1, thus an 8-word LUT was used, with a precision of 16 bits.

<!-- <div align="center">
<img src="images/gelu.drawio.png" alt="Implemented Model" width="450" height="325">
 <p>High-Level Flow Diagram of GELU AF</p>
</div> -->

![MAC](https://raw.githubusercontent.com/DeepWok/mase/main/docs/source/imgs/hardware/activations/gelu.drawio.png)