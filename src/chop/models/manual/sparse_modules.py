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
        self.gradient = torch.ones_like(self.weight).flatten().to(self.weight.device)
        self.index_map = {
            "magnitude": torch.abs,
            "gradient": lambda input: torch.abs(input) * self.gradient,
        }

        self.disable_adapter = False
        self.disable_sparse = False

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
        else:
            pass

        if init_sparse_weights:
            self.reset_sparse_parameters(adapter_name)

        self.to(self.weight.device)

    def reset_sparse_parameters(self, adapter_name):
        if adapter_name in self.sparse_train.keys():
            nn.init.eye_(self.sparse_train[adapter_name].weight)


class LinearSparse(nn.Linear, SparseLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: dict = None,
        **kwargs,
    ):
        self.config = config
        init_sparse_weights = self.config.get("init_sparse_weights", True)

        (
            k,
            sparse_alpha,
            sparse_dropout,
            idx_method,
            adapter_name,
            disable_adapter,
        ) = (
            config["k"],
            config["sparse_alpha"],
            config["sparse_dropout"],
            config["idx_method"],
            config["adapter_name"],
            config["disable_adapter"],
        )
        sparse_dropout = float(sparse_dropout)
        self.idx_method = idx_method

        # determining indexing
        self.step = 0
        self.idx = 0
        self.selected_weights = 0

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        SparseLayer.__init__(self)

        self.disable_adapter = disable_adapter
        self.weight.requires_grad = False
        self.zero_tensor = torch.zeros(size=self.weight.shape).flatten()
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

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(
            input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias
        )

    def update_weight_selection(self, k):
        w_flat = self.weight.flatten()
        _, self.idx = torch.topk(
            self.index_method(w_flat, self.idx_method),
            k,
            sorted=True,
        )
        self.selected_weights = torch.gather(w_flat, dim=0, index=self.idx)

        return self.selected_weights, self.idx

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.sparse_train.keys():
            return self._linear(x)

        if self.disable_adapter or self.disable_sparse:
            return self._linear(x)

        elif self.k[self.active_adapter] > 0:
            sparse = self.sparse_train[self.active_adapter]
            dropout = self.sparse_dropout[self.active_adapter]
            k = self.k[self.active_adapter]
            scaling = self.scaling[self.active_adapter]
            # Top-k selection

            if self.step == 0:
                self.selected_weights, self.idx = self.update_weight_selection(k)

            # Apply sparse adapter
            output = sparse(self.selected_weights)
            scaled_output = output * scaling

            # Scatter adapted values into weight tensor
            adapted_weights = torch.scatter(
                self.zero_tensor.to(x.device),
                dim=0,
                index=self.idx,
                src=scaled_output,
            ).view(self.unflattened_size)

            self.step += 1
            new_weight = torch.add(self.weight, adapted_weights)
            x = x.to(sparse.weight.dtype)

            result = F.linear(
                dropout(x),
                transpose(new_weight, self.fan_in_fan_out),
                bias=self.bias,
            )

        else:
            result = self._linear(x)

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
