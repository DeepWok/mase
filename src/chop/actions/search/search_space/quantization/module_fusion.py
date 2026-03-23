"""
Search space that jointly searches over quantisation bit-widths and kernel fusion
strategy (FlexAttention + fused RMSNorm).

Works directly on nn.Module (no FX tracing required), making it compatible with
HuggingFace models such as BERT, TinyLlama, and Mistral.
"""
from copy import deepcopy

import torch
from torch import nn

from ..base import SearchSpaceBase
from .....passes.module.transforms.quantize.quantize import (
    quantize_module_transform_pass,
)
from .....passes.module.transforms.attention.flex_attention_transform import (
    flex_attention_transform_pass,
)


# Choices exposed to Optuna for each quantisation dimension
_QUANT_CHOICES = {
    "data_in_width": [4, 8, 16, 32],
    "weight_width": [4, 8, 16, 32],
}

# Fusion strategies available.  "fused_rmsnorm" and "both" are silently skipped
# if Part 2's pass is not yet merged into this branch.
_FUSION_STRATEGIES = ["none", "flex_attention", "fused_rmsnorm", "both"]


class ModuleSearchSpaceQuantizationFusion(SearchSpaceBase):
    """
    Post-training quantisation + kernel fusion search space operating on plain nn.Module.

    Search dimensions:
      - data_in_width  : activation bit-width  {4, 8, 16, 32}
      - weight_width   : weight bit-width       {4, 8, 16, 32}
      - fusion_strategy: kernel fusion applied  {none, flex_attention, fused_rmsnorm, both}

    The TOML config block for this space looks like::

        [search.search_space]
        name = "module/quantize_fusion"

        [search.search_space.setup]
        score_mod = "causal"          # score_mod passed to flex_attention_transform_pass
        score_mod_kwargs = {}         # optional kwargs (e.g. window_size for sliding_window)

        [search.search_space.seed.default]
        data_in_width    = [4, 8, 16, 32]
        weight_width     = [4, 8, 16, 32]
        fusion_strategy  = ["none", "flex_attention", "fused_rmsnorm", "both"]
    """

    def _post_init_setup(self) -> None:
        self.model.to("cpu")
        setup = self.config.get("setup", {})
        self.score_mod = setup.get("score_mod", "causal")
        self.score_mod_kwargs = setup.get("score_mod_kwargs", {})

        # Read per-dimension choices from seed config, falling back to defaults
        seed = self.config.get("seed", {}).get("default", {})
        self._quant_choices = {
            k: seed.get(k, defaults)
            for k, defaults in _QUANT_CHOICES.items()
        }
        self._fusion_choices = seed.get("fusion_strategy", _FUSION_STRATEGIES)

    # ------------------------------------------------------------------
    # SearchSpaceBase interface
    # ------------------------------------------------------------------

    def build_search_space(self) -> None:
        self.choices_flattened = {}
        self.choice_lengths_flattened = {}

        for name, choices in self._quant_choices.items():
            self.choices_flattened[name] = choices
            self.choice_lengths_flattened[name] = len(choices)

        self.choices_flattened["fusion_strategy"] = self._fusion_choices
        self.choice_lengths_flattened["fusion_strategy"] = len(self._fusion_choices)

    def flattened_indexes_to_config(self, indexes: dict[str, int]) -> dict:
        """Convert Optuna integer indexes → human-readable config dict."""
        quant_config = {
            name: self.choices_flattened[name][indexes[name]]
            for name in self._quant_choices
        }
        fusion_strategy = self._fusion_choices[indexes["fusion_strategy"]]
        return {
            "quantization": quant_config,
            "fusion_strategy": fusion_strategy,
        }

    def rebuild_model(self, sampled_config: dict, is_eval_mode: bool = True) -> nn.Module:
        """
        Deep-copy the base model and apply:
          1. quantize_module_transform_pass  (always)
          2. flex_attention_transform_pass   (if fusion_strategy in {flex_attention, both})
          3. rmsnorm_residual_fusion_pass    (if fusion_strategy in {fused_rmsnorm, both}
                                             and Part 2 is merged)
        """
        model = deepcopy(self.model).to(self.accelerator)
        model.eval() if is_eval_mode else model.train()

        if sampled_config is None:
            return model

        # 1. Quantisation — apply uniform bit-width to all Linear layers by type
        quant_cfg = sampled_config["quantization"]
        quant_pass_args = {
            "by": "type",
            "linear": {
                "config": {
                    "name": "integer",
                    "data_in_width": quant_cfg["data_in_width"],
                    "data_in_frac_width": quant_cfg["data_in_width"] // 2,
                    "weight_width": quant_cfg["weight_width"],
                    "weight_frac_width": quant_cfg["weight_width"] // 2,
                    "bias_width": quant_cfg["weight_width"],
                    "bias_frac_width": quant_cfg["weight_width"] // 2,
                }
            },
        }
        model, _ = quantize_module_transform_pass(model, quant_pass_args)

        # 2. FlexAttention fusion
        strategy = sampled_config["fusion_strategy"]
        if strategy in ("flex_attention", "both"):
            flex_args = {
                "score_mod": self.score_mod,
                "score_mod_kwargs": self.score_mod_kwargs,
            }
            model, _ = flex_attention_transform_pass(model, flex_args)

        # 3. Fused RMSNorm + residual (Part 2) — imported lazily so the search
        #    space works before Part 2 is merged into this branch.
        if strategy in ("fused_rmsnorm", "both"):
            try:
                from .....passes.module.transforms.fused_ops.rmsnorm_residual_fusion import (
                    rmsnorm_residual_fusion_pass,
                )
                model, _ = rmsnorm_residual_fusion_pass(model, {})
            except ImportError:
                pass  # Part 2 not yet available — skip silently

        return model
