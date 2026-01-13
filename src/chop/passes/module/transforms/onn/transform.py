try:
    import mase_triton

    MASE_TRITON_IS_AVAILABLE = True
except ImportError:
    MASE_TRITON_IS_AVAILABLE = False

from typing import TypedDict

import torch
from transformers.models.llama.modeling_llama import LlamaAttention as HfLlamaAttention

from ...module_modify_helper import replace_by_name
from ...state_dict_map import match_a_pattern
from .layers.attn import OtLlamaAttention
from .layers.linear import OtLinear


def get_config_by_name(config: dict, name: str):
    if name in config:
        return config[name]
    else:
        if "default" in config:
            return config["default"]
        else:
            return None


def get_config_by_regex_name(config: dict, name: str):
    matched_pattern = match_a_pattern(name, config.keys())
    if matched_pattern is None:
        if "default" in config:
            return config["default"]
        else:
            return None
    else:
        return config[matched_pattern]


def get_layer_config(
    layer_name_to_config: dict[str, dict], use_regex: bool, layer_name: str
) -> dict | None:
    if use_regex:
        config = get_config_by_regex_name(layer_name_to_config, layer_name)
    else:
        config = get_config_by_name(layer_name_to_config, layer_name)
    return config


class OtTransformConfig(TypedDict):
    q_levels: int
    q_lut_min: float
    q_smooth_factor: float
    q_init_seed: int
    q_bypass: bool

    @classmethod
    def create_default(cls) -> "OtTransformConfig":
        return cls(
            q_levels=256,
            q_lut_min=0.020040,
            q_smooth_factor=0.9,
            q_init_seed=0,
            q_bypass=False,
        )


if MASE_TRITON_IS_AVAILABLE:
    _SUPPORTED_MODULE_CLS = (torch.nn.Linear, HfLlamaAttention)

    def optical_transformer_module_transform_pass(
        network: torch.nn.Module, pass_args: dict
    ) -> torch.nn.Module:
        """
        Transform a neural network by replacing supported modules with their optical transformer equivalents.

        This pass simulates optical neural network (ONN) computation by replacing standard PyTorch
        modules with quantized optical transformer layers. The optical transformer model is based on
        the `Optical Transformers paper <https://arxiv.org/abs/2302.10360>`_.

        Supported module replacements:

        - ``torch.nn.Linear`` → ``OtLinear``
        - ``transformers.models.llama.modeling_llama.LlamaAttention`` → ``OtLlamaAttention``

        Args:
            network (torch.nn.Module): The input network to be transformed.
            pass_args (dict): A dictionary containing transformation configurations.

                - ``by`` (str): Layer matching strategy. Either ``'name'`` for exact name matching
                  or ``'regex_name'`` for regex-based pattern matching. Defaults to ``'regex_name'``.
                - ``default`` (dict, optional): Default configuration applied to all matching layers.
                - ``<layer_name_or_pattern>`` (dict): Per-layer configuration. Each layer config
                  can contain the following keys:

                  - ``q_levels`` (int): Number of quantization levels. Default: 256.
                  - ``q_lut_min`` (float): Minimum value for lookup table. Default: 0.020040.
                  - ``q_smooth_factor`` (float): Smoothing factor for running statistics. Default: 0.9.
                  - ``q_init_seed`` (int): Random seed for quantization initialization. Default: 0.
                  - ``q_bypass`` (bool): If True, bypass optical quantization. Default: False.

        Returns:
            torch.nn.Module: The transformed network with optical transformer modules.

        Raises:
            RuntimeError: If ``mase-triton`` is not installed.

        Example:
            .. code-block:: python

                from chop.passes.module.transforms.onn import optical_transformer_module_transform_pass

                # Transform all linear layers with default config
                pass_args = {
                    "by": "regex_name",
                    "default": {
                        "q_levels": 256,
                        "q_lut_min": 0.020040,
                        "q_bypass": False,
                    }
                }
                transformed_model = optical_transformer_module_transform_pass(model, pass_args)

        Note:
            This pass requires the ``mase-triton`` package to be installed.
            Install via ``pip install mase-triton``.
        """
        by = pass_args.pop("by", "regex_name")
        assert by in [
            "name",
            "regex_name",
        ], f"`by` can be either 'name' or 'regex_name', but got {by}"
        # replace attn layers if any
        for m_name, m in network.named_modules():
            if not isinstance(m, HfLlamaAttention):
                continue
            m_config = get_layer_config(
                pass_args, use_regex=by == "regex_name", layer_name=m_name
            )
            if m_config is None:
                continue
            if isinstance(m, HfLlamaAttention):
                new_m = OtLlamaAttention.from_pretrained(m, **m_config)
            elif isinstance(m, _SUPPORTED_MODULE_CLS):
                continue
            else:
                raise NotImplementedError(
                    f"ONN transform for type {type(m)} is supported"
                )
            replace_by_name(network, name=m_name, module=new_m)
        # replace linear layers if any
        for m_name, m in network.named_modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            m_config = get_layer_config(
                pass_args, use_regex=by == "regex_name", layer_name=m_name
            )
            if m_config is None:
                continue
            if isinstance(m, torch.nn.Linear):
                new_m = OtLinear.from_linear(m, **m_config)
            elif isinstance(m, _SUPPORTED_MODULE_CLS):
                continue
            else:
                raise NotImplementedError(
                    f"ONN transform for type {type(m)} is supported"
                )
            replace_by_name(network, name=m_name, module=new_m)
        return network

else:

    def optical_transformer_module_transform_pass(
        network: torch.nn.Module, pass_args: dict
    ) -> torch.nn.Module:
        raise RuntimeError(
            "`mase-triton` is needed for ONN transform. Install via `pip install mase-triton`."
        )
