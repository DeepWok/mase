import math
import re
import warnings
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import List, Optional, Tuple, Union
from .lora_utils import transpose
import torch
import torch.nn as nn
import torch.nn.functional as F

# from transformers.pytorch_utils import Conv1

# LoraLayer & Linear copied from https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py#L675


class LoraLayer:
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapter = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs
        init_lora_weights = bool(field(default=True))

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights
    ):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if self.disable_adapter == False:
            if r > 0:
                self.lora_A.update(
                    nn.ModuleDict(
                        {adapter_name: nn.Linear(self.in_features, r, bias=False)}
                    )
                )
                self.lora_B.update(
                    nn.ModuleDict(
                        {adapter_name: nn.Linear(r, self.out_features, bias=False)}
                    )
                )
                self.scaling[adapter_name] = lora_alpha / r
        else:
            pass

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])


class LinearLora(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: dict = None,
        **kwargs,
    ):
        self.config = config
        init_lora_weights = self.config.get("init_lora_weights", True)

        r, lora_alpha, lora_dropout, adapter_name, disable_adapter = (
            config["r"],
            config["lora_alpha"],
            config["lora_dropout"],
            config["adapter_name"],
            config["disable_adapter"],
        )
        lora_dropout = float(lora_dropout)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.disable_adapter = disable_adapter
        self.fan_in_fan_out = config.get("fan_in_fan_out", False)
        self.is_target_conv_1d_layer = config.get("is_target_conv_1d_layer", False)

        if self.fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
        self.is_target_conv_1d_layer = self.is_target_conv_1d_layer

    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += self.get_delta_weight(self.active_adapter)
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= self.get_delta_weight(self.active_adapter)
            self.merged = False

    def get_delta_weight(self, adapter):
        return (
            transpose(
                self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                self.fan_in_fan_out,
            )
            * self.scaling[adapter]
        )

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
        )

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.lora_A.keys():
            return self._linear(x)

        if self.disable_adapter:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = self._linear(x)

        elif self.r[self.active_adapter] == 0 or self.merged:
            result = self._linear(x)

        else:
            lora_A = self.lora_A[self.active_adapter]
            lora_B = self.lora_B[self.active_adapter]
            dropout = self.lora_dropout[self.active_adapter]
            scaling = self.scaling[self.active_adapter]

            result = self._linear(x)
            x = x.to(lora_A.weight.dtype)
            result += lora_B(lora_A(dropout(x))) * scaling

        result = result.to(previous_dtype)

        return result

    def extract_lora_params(self):
        lora_params = {
            "lora_A": self.lora_A[self.active_adapter].state_dict(),
            "lora_B": self.lora_B[self.active_adapter].state_dict(),
        }

        return lora_params

    # Helper function to bias the training towards either the target module or the entire model


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    # Paramter: bias -> Which modules should be marked as trainable based on the given options
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return model
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError
