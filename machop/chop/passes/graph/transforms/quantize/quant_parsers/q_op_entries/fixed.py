FIXED_OP_ENTRIES = {
    "add": {
        "required": ("name", "data_in_width", "data_in_frac_width"),
        "optional": ("bypass",),
    },
    "bmm": {
        "required": (
            "name",
            "data_in_width",
            "data_in_frac_width",
            "weight_width",
            "weight_frac_width",
        ),
        "optional": ("bypass",),
    },
    "conv1d": {
        "required": (
            "name",
            "data_in_width",
            "data_in_frac_width",
            "weight_width",
            "weight_frac_width",
        ),
        "optional": ("bypass", "bias_width", "bias_frac_width"),
    },
    "conv2d": {
        "required": (
            "name",
            "data_in_width",
            "data_in_frac_width",
            "weight_width",
            "weight_frac_width",
        ),
        "optional": ("bypass", "bias_width", "bias_frac_width"),
    },
    "linear": {
        "required": (
            "name",
            "data_in_width",
            "data_in_frac_width",
            "weight_width",
            "weight_frac_width",
        ),
        "optional": ("bypass", "bias_width", "bias_frac_width"),
    },
    "matmul": {
        "required": (
            "name",
            "data_in_width",
            "data_in_frac_width",
            "weight_width",
            "weight_frac_width",
        ),
        "optional": ("bypass",),
    },
    "relu": {
        "required": ("name", "data_in_width", "data_in_frac_width"),
        "optional": ("bypass",),
    },
    "sub": {
        "required": ("name", "data_in_width", "data_in_frac_width"),
        "optional": ("bypass",),
    },
    "rotary_positional_encoding": {
        "required": ("name", "data_in_width", "data_in_frac_width"),
        "optional": ("bypass",),
    },
}
