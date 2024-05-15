# Quantizers

## Software-simulated quantization

| name | paper | model@dataset | blocking dimension | representation | extra |
| --- | --- | --- | --- | --- | --- |
| `integer_quantizer(x, width, frac_width)` | - | - | - | $(i/2^{f})$, <br> signed int $i$, the number of fractional bits $f$ | signed fixed-point number |
| `minifloat_denorm_quantizer(x, width, exponent_width, exponent_bias)` | - | - | - | $(-1)^s 2^e m$, <br> exponent $e$, mantissa $m$ | no implicit leading bit in mantissa|
| `minifloat_ieee_quantizer(x, width, exponent_width, exponent_bias)` | - | - | - | $(-1)^s 2^e m'$, <br> normal: $m'=1.0+m$, subnormal: $m'=m$ | an implicit leading bit in mantissa |
| `log_quantizer(x, width, exponent_bias)` | [CNNs using Logarithmic Data Representation](http://arxiv.org/abs/1603.01025) | VGG16@CIFAR10, ALEXNET@CIFAR10 | - | $(-1)^s 2^e$ | - |
| `msfp_quatizer(x, width, exponent_width, exponent_bias, block_size)`| [Microsoft MSFP](https://proceedings.neurips.cc/paper/2020/hash/747e32ab0fea7fbd2ad9ec03daa3f840-Abstract.html) | CNNs, RNNs, <br> Transformers (BERT@MRPC, BERT@SQuAD1.1, BERT@SQuADv2) | Linear matrix: tiles along matrix row, <br> Conv2D: tiles along channel depth | $2^{e_{shared}}[(-1)^{s_1} m_1, (-1)^{s_2} m_2, \dots]$ |
| `block_minifloat_quantizer(x, width, exponent_width, bias_width, block_size)` | [Philip Leong's Block Minifloat](https://openreview.net/forum?id=6zaTwpNSsQ2) | CNNs, RNNs, Transformer (Transformer-base@IWSLT) | Matrix Multiply: $N\times N$ square block. <br> Conv2D (?) | $2^{-b_{shared}}[(-1)^{s_1} 2^{e'_1}m'_1, (-1)^{s_2}2^{e'_2}m'_2, \dots]$,  <br> the shared exponent bias:$b_{shared}$|  both forward and backward uses software-simulated quantized values|
| `block_log_quantizer(x, width, exponent_bias_width, block_size)` | - | - | - | $2^{-b_{shared}}[(-1)^{s_1} 2^{e'_1}, (-1)^{s_2}2^{e'_2}, \dots]$, <br> the shared exponent bias $b_{shared}$ |  |

The following quantizers are not supported yet

| name | paper | model@dataset | blocking dimension | representation | extra |
| --- | --- | --- | --- | --- | --- |
| ⬜ TODO: `mx_quantizer` | [Microsoft's MX](https://arxiv.org/abs/2302.08007) | See Table III in the paper. CNNs, RNNs, ViT (DeiT-Tiny/-Small@ImageNet), <br> Transformer (Transformer-base/-large@WMT-17, BERT-base/-large@Wikipedia, GPT-XS/-S/-M/-L/-XL@?) | Two-level scaling on vectors | $2^{e_{s}}\Big[ 2^{e_{ss_1}} [(-1)^{s_1} m_1, (-1)^{s_2} m_2  ], 2^{e_{ss_2}}[(-1)^{s_3} m_3, (-1)^{s_3} m_3 ]\Big]$ | no implicit leading bit in mantissa  |

## Two-level block quantization for large language models

Large language models requires significant GPU resources for inference. One way to reduce the inference resource consumption is quantization. The challenge of quantizing models is the significant accuracy degradation as the bit-width decreases. To solve this accuracy degradation, block-based number formats have been proposed. One or two levels of scaling factors are shared over a block of numbers, where the scaling factor can be exponent, exponent bias, mantissa, fixed-point number, or floating-point number. A proper blocking and sharing scheme mitigates the impact of extreme outlier values. However, block-based quantization of large language models remains to be explored, especially low bit width (1-bit/2-bit) quantization. Here we aim to explore different combinations of shared components on large language models. Specifically, we aim to estimate the hardware cost of each combination, and compare corresponding accuracy degradation given the same block size.

💡In `Facebook/OPT-350m` for language modeling, the Linear layers take up `168.02G / 174.68G x 100%=96.19%` FLOPs.
