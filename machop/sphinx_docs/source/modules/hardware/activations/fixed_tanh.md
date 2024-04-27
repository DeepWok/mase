# [Tanh](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html)

Tanh is defined as:

`Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`

This function is applied element-wise.

## Implementation

Hyperbolic tangent is an odd function:

tanh(-x) = -tanh(x)

Using this property, only the absolute value of the input needs to be processed, and the input sign can be directly passed to the output.

Piecewise quadratic approximation is used for tanh activation function as detailed in [3]. In order to preserve the continuity property of the first-order derivative, [3] approximates the first-order derivative of tanh function using piecewise linear approximations. The approximation of the tanh function is then obtained by integrating the approximation of the first-order derivative. The resulting approximation is provided in the following equation, where $m_1$, $m_2$, $c_1$, $c_2$, $a$, $b$, $d_1$ and $d_2$ are $-0.54324$, $-0.16957$, $1$, $0.42654$, $1.52$, $2.57$, $0.016$ and $0.4519$ respectively.

<!-- <div align="center">
<img src="images/eqn2.png" alt="Implemented Model" width="450" height="80">
</div> -->

![MAC](https://raw.githubusercontent.com/DeepWok/mase/main/machop/sphinx_docs/source/imgs/hardware/activations/eqn2.png)

The high-level flow diagram for tanh AF implementation is given in the following figure.

<!-- <div align="center">
<img src="images/tanh_fd.png" alt="Implemented Model" width="450" height="350">
 <p>High-Level Flow Diagram of Tanh AF</p>
</div> -->

![MAC](https://raw.githubusercontent.com/DeepWok/mase/main/machop/sphinx_docs/source/imgs/hardware/activations/tanh_fd.png)

## References

1. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
4. Chandra, M., 2021. On the Implementation of Fixed-Point Exponential Function for Machine Learning and Signal-Processing Accelerators. IEEE Design & Test, 39(4), pp.64-70.
5. Lin, C.W. and Wang, J.S., 2008, May. A digital circuit design of hyperbolic tangent sigmoid function for neural networks. In 2008 IEEE International Symposium on Circuits and Systems (ISCAS) (pp. 856-859). IEEE.
6. González-Díaz_Conti, G., Vázquez-Castillo, J., Longoria-Gandara, O., Castillo-Atoche, A., Carrasco-Alvarez, R., Espinoza-Ruiz, A. and Ruiz-Ibarra, E., 2021. Hardware-based activation function-core for neural network implementations. Electronics, 11(1), p.14.