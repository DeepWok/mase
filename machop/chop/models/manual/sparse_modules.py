import math
import re
import warnings
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import List, Optional, Tuple, Union
from .sparse_utils import transpose
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseLayer:
    def __init__(self, **kwargs):
        self.k = {}
        self.sparse_alpha = {}
        self.scaling = {}
        self.sparse_dropout = nn.ModuleDict({})
        self.sparse_train = nn.ModuleDict({})
        self.index_map = {
            "magnitude": torch.abs,
        }
        self.kwargs = kwargs
        init_sparse_weights = bool(field(default=True))

    def update_layer(
        self, adapter_name, k, sparse_alpha, sparse_dropout, init_sparse_weights
    ):
        if 0 <= k < 1:
            self.k[adapter_name] = math.ceil(k * self.weight.nelement())
        elif k >= 1:
            self.k[adapter_name] = k
        else:
            raise ValueError(f"Invalid k number: {k}. k must be > 0")

        self.sparse_alpha[adapter_name] = sparse_alpha

        if sparse_dropout > 0.0:
            sparse_dropout_layer = nn.Dropout(p=sparse_dropout)
        else:
            sparse_dropout_layer = nn.Identity()

        self.sparse_dropout.update(nn.ModuleDict({adapter_name: sparse_dropout_layer}))

        # Actual trainable parameters
        if k > 0:
            self.sparse_train.update(
                nn.ModuleDict(
                    {
                        adapter_name: nn.Linear(
                            self.k[adapter_name], self.k[adapter_name], bias=False
                        )
                    }
                )
            )
            self.scaling[adapter_name] = sparse_alpha

        if init_sparse_weights:
            self.reset_sparse_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_sparse_parameters(self, adapter_name):
        if adapter_name in self.sparse_train.keys():
            nn.init.kaiming_uniform_(
                self.sparse_train[adapter_name].weight, a=math.sqrt(5)
            )


class LinearSparse(nn.Linear, SparseLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: dict = None,
        **kwargs,
    ):
        init_sparse_weights = kwargs.pop("init_sparse_weights", False)
        self.config = config

        (k, sparse_alpha, sparse_dropout, idx_method, adapter_name) = (
            config["k"],
            config["sparse_alpha"],
            config["sparse_dropout"],
            config["idx_method"],
            config["adapter_name"],
        )
        sparse_dropout = float(sparse_dropout)

        self.idx_method = idx_method

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        SparseLayer.__init__(self)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.zero_tensor = torch.zeros(size=self.weight.shape)
        self.unflattened_size = tuple(self.weight.shape)

        self.fan_in_fan_out = config.get("fan_in_fan_out", False)

        if self.fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(
            adapter_name, k, sparse_alpha, sparse_dropout, init_sparse_weights
        )
        self.active_adapter = adapter_name

    def index_method(self, tensor, idx_key):
        # Retrieve the method based on the user input
        selected_method = self.index_map.get(idx_key, None)
        if selected_method is None:
            raise ValueError(f"Unknown method_key: {idx_key}")

        return selected_method(tensor)

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.sparse_train.keys():
            return F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )

        elif self.k[self.active_adapter] > 0:
            x = x.to(self.sparse_train[self.active_adapter].weight.dtype)

            # Top-k selection
            w_flat = self.weight.flatten()
            val, idx = torch.topk(
                self.index_method(w_flat, self.idx_method),
                self.k[self.active_adapter],
                sorted=True,
            )
            selected_weights = torch.gather(w_flat, dim=0, index=idx)

            # Apply sparse adapter
            adapted_output = self.sparse_train[self.active_adapter](selected_weights)
            scaled_output = adapted_output * self.scaling[self.active_adapter]

            # Scatter adapted values into weight tensor
            adapted_weights = torch.scatter(
                self.zero_tensor.flatten().to(x.device),
                dim=0,
                index=idx,
                src=scaled_output,
            ).view(self.unflattened_size)
            new_weight = torch.add(self.weight, adapted_weights)

            result = F.linear(
                self.sparse_dropout[self.active_adapter](x),
                transpose(new_weight, self.fan_in_fan_out),
                bias=self.bias,
            )

        else:
            result = F.linear(
                x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
            )

        result = result.to(previous_dtype)

        return result

    def extract_sparse_params(self):
        sparse_params = {
            "sparse_train": self.sparse_train[self.active_adapter].state_dict(),
        }

        return sparse_params

    # Helper function to bias the training towards either the target module or the entire model


def mark_only_sparse_as_trainable(model: nn.Module, bias: str = "none") -> None:
    # Paramter: bias -> Which modules should be marked as trainable based on the given options
    for n, p in model.named_parameters():
        if "sparse_" not in n:
            p.requires_grad = False
    if bias == "none":
        return model
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "sparse_only":
        for m in model.modules():
            if isinstance(m, SparseLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError
