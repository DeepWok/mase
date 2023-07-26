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
        self.disable_adapters = False
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
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_conv2d(
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
        if r > 0:
            kernel_size = self.kwargs["kernel_size"]
            stride = self.kwargs["stride"]
            padding = self.kwargs["padding"]
            self.lora_A.update(
                nn.ModuleDict(
                    {
                        adapter_name: nn.Conv2d(
                            self.in_features,
                            r,
                            kernel_size,
                            stride,
                            padding,
                            bias=False,
                        )
                    }
                )
            )
            self.lora_B.update(
                nn.ModuleDict(
                    {
                        adapter_name: nn.Conv2d(
                            r, self.out_features, (1, 1), (1, 1), bias=False
                        )
                    }
                )
            )
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def update_layer_embedding(
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
        if r > 0:
            weight_A = torch.randn(
                (r, self.in_features),
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            weight_B = torch.randn(
                (self.out_features, r),
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            self.lora_embedding_A.update(
                nn.ParameterDict({adapter_name: nn.Parameter(weight_A)})
            )
            self.lora_embedding_B.update(
                nn.ParameterDict({adapter_name: nn.Parameter(weight_B)})
            )
            self.scaling[adapter_name] = lora_alpha / r
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


class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        config: dict = None,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.config = config

        r, lora_alpha, lora_dropout = (
            config["r"],
            config["lora_alpha"],
            config["lora_dropout"],
        )
        lora_dropout = float(
            lora_dropout
        )  # original huggingface implementation requires float
        # self.r = r
        # self.lora_alpha = lora_alpha
        # self.lora_dropout = lora_dropout

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

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

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_A.keys():
            return F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )
        elif self.r[self.active_adapter] > 0 and not self.merged:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )

            x = x.to(self.lora_A[self.active_adapter].weight.dtype)

            result += (
                self.lora_B[self.active_adapter](
                    self.lora_A[self.active_adapter](
                        self.lora_dropout[self.active_adapter](x)
                    )
                )
                * self.scaling[self.active_adapter]
            )
        else:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )

        result = result.to(previous_dtype)

        return result

    def extract_lora_params(self):
        lora_params = {
            "lora_A": self.lora_A[self.active_adapter].state_dict(),
            "lora_B": self.lora_B[self.active_adapter].state_dict(),
        }

        return lora_params
