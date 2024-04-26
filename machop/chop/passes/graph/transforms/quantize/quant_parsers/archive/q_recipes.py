from dataclasses import dataclass, field


@dataclass
class QRecipeFixed:
    """_summary_
    Fixed point quantization
    """

    name: str = field(default="fixed", init=False)
    bypass: bool = field(default=False)
    data_in_width: int
    data_in_frac_width: int
    weight_width: int | None = field(default=None)
    weight_frac_width: int | None = field(default=None)
    bias_width: int | None = field(default=None)
    bias_frac_width: int | None = field(default=None)


@dataclass
class QRecipeLutNet:
    """
    LUTNET quantization

    binarization_level (int): which level of binarization is applied, "binarized_weight" is only weights binarized others is no binarization
    input_expanded (bool): If set to True, means all LUT's inputs are considered during calculations , else only the first input will considered and the remaining will be masked.
    k: int  # k entries of a LUT
    levels (int): number of residual levels to use in LUTNET
    dim: this is needed by convolution
    """

    name: str = field(default="lutnet", init=False)

    data_in_width: int
    data_in_frac_width: int
    data_in_binarization_level: int
    data_in_input_expanded: bool
    data_in_k: int
    data_in_in_levels: int
    data_in_dim: tuple[int]

    weight_width: int
    weight_frac_width: int
    weight_binarization_level: int
    weight_input_expanded: bool
    weight_k: int
    weight_in_dim: tuple[int]

    bias_width: int
    bias_frac_width: int
    bias_binarization_level: int
    bias_input_expanded: bool
    bias_k: int
    bias_in_dim: tuple[int]
