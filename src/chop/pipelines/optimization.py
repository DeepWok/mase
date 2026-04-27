"""
OptimizationPipeline: applies quantisation + kernel fusion passes to a model
in a single call given a config dict.

Used for:
  - Evaluating a specific best config found by the Optuna search
  - Running the ablation table in the notebook
  - Standalone benchmarking without the search loop

Example::

    from chop.pipelines.optimization import OptimizationPipeline

    pipeline = OptimizationPipeline(model, score_mod="causal")
    optimized = pipeline.run({
        "quantization": {"data_in_width": 8, "weight_width": 8},
        "fusion_strategy": "both",
    })
"""
from copy import deepcopy

import torch
from torch import nn

from ..passes.module.transforms.quantize.quantize import quantize_module_transform_pass
from ..passes.module.transforms.attention.flex_attention_transform import (
    flex_attention_transform_pass,
)


class OptimizationPipeline:
    """
    Applies quantisation and kernel fusion passes to a model.

    Parameters
    ----------
    model : nn.Module
        Base (unoptimised) model.  It is never mutated; each call to
        :meth:`run` works on a deep copy.
    score_mod : str
        Score modification function name passed to FlexAttention
        (e.g. ``"causal"``, ``"sliding_window"``, ``"alibi"``).
    score_mod_kwargs : dict, optional
        Extra keyword arguments for parameterised score mods
        (e.g. ``{"window_size": 4096}`` for ``sliding_window``).
    device : str or torch.device, optional
        Target device.  Defaults to ``"cuda"`` if available, else ``"cpu"``.
    """

    def __init__(
        self,
        model: nn.Module,
        score_mod: str = "causal",
        score_mod_kwargs: dict | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.score_mod = score_mod
        self.score_mod_kwargs = score_mod_kwargs or {}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def run(self, config: dict) -> nn.Module:
        """
        Return an optimised copy of the base model.

        Parameters
        ----------
        config : dict
            Must contain:

            ``quantization`` : dict
                Keys: ``data_in_width``, ``weight_width`` (int bit-widths).
            ``fusion_strategy`` : str
                One of ``"none"``, ``"flex_attention"``,
                ``"fused_rmsnorm"``, ``"both"``.

        Returns
        -------
        nn.Module
            Deep copy of the base model with the requested passes applied.
        """
        model = deepcopy(self.model).to(self.device)
        model.eval()

        # 1. Quantisation
        quant_cfg = config["quantization"]
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

        # 2. FlexAttention
        strategy = config["fusion_strategy"]
        if strategy in ("flex_attention", "both"):
            model, _ = flex_attention_transform_pass(
                model,
                {"score_mod": self.score_mod, "score_mod_kwargs": self.score_mod_kwargs},
            )

        # 3. Fused RMSNorm + residual (Part 2) — imported lazily
        if strategy in ("fused_rmsnorm", "both"):
            try:
                from ..passes.module.transforms.fused_ops.rmsnorm_residual_fusion import (
                    rmsnorm_residual_fusion_pass,
                )
                model, _ = rmsnorm_residual_fusion_pass(model, {})
            except ImportError:
                pass  # Part 2 not yet merged — skip silently

        return model
