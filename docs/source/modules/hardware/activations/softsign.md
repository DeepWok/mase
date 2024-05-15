# [SoftSign](https://pytorch.org/docs/stable/generated/torch.nn.Softsign.html)

Applies the element-wise function:

`SoftSign(x) = x / (1 + |x|)`

## Implementation

In order to avoid the hardware cost associated with division, softsign function was approximated with piecewise quadratic functions. It should be noted that softsign function is odd. Furthermore, limx→∞ = 1 and limx→−∞ = −1.

Those properties allow us to only approximate the range of inputs (0,16)- negative inputs are sign inverted for calculation and then converted back to negative, while numbers larger than 16 are approximated to 1. For values that need piecewise approximation, a 32 word LUT is used, to approximate the function in intervals of 0.5. The precision was chosen to be 12 bits, with 11 fractional bits.

 The number of approximations, values of coefficients and quantisation were determined using a custom Python script. The code calculated the optimal coefficients for a given number of approximations and simulated approximating the function for a given quantisation choice.

<!-- <div align="center">
<img src="images/softsign.drawio.png" alt="Implemented Model" width="450" height="350">
 <p>High-Level Flow Diagram of Softsign AF</p>
</div> -->

![MAC](https://raw.githubusercontent.com/DeepWok/mase/main/docs/source/imgs/hardware/activations/softsign.drawio.png)