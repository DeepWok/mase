from collections import defaultdict
from copy import deepcopy
from accelerate import (
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
from collections.abc import MutableMapping

# from chop.passes.transforms.quantize import QUANTIZEABLE_OP, quantize_transform_pass
# from chop.models.manual.opt_quantized import name_hash as opt_name_hash

# get llm specific funcs
from .quan_config_opt import parse_opt_quantized_config
from .sampler_opt import sample_opt_quant_config
from ..base import SearchSpaceBase


def flatten_dict(nested_dict, parent_key=None, flattened={}, separator="/"):
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key is not None else key
        if isinstance(value, dict):  # If the value is another nested dictionary
            flatten_dict(value, new_key, flattened)
        else:  # If the value is a leaf node
            flattened[new_key] = value
    return flattened


def unflatten_dict(flat_dict, separator="/"):
    nested_dict = {}
    for key, value in flat_dict.items():
        keys = key.split(separator)
        current_dict = nested_dict
        for i, k in enumerate(keys[:-1]):
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        current_dict[keys[-1]] = value
    return nested_dict


class LLMMixedPrecisionSearchSpace(SearchSpaceBase):
    default_config = {
        "name": "integer",
        "bias_frac_width": 5,
        "bias_width": 8,
        "data_in_frac_width": 5,
        "data_in_width": 8,
        "weight_frac_width": 3,
        "weight_width": 8,
    }

    def get_num_layers(self):
        return self.model["model"].config.num_hidden_layers

    def build_search_space(self):
        self.config_parser = parse_opt_quantized_config
        self.config_sampler = sample_opt_quant_config
        self.quan_by = self.config["quantization"].pop("by", "name")
        self.quan_config_seed = self.config["quantization"]

    def build_opt_seed_and_indexes(self):
        config_seed, indexes, index_ranges = {}, {}, {}
        zero_seed = {k: 0 for k, v in self.quan_config_seed.items()}
        ranges = {k: len(v) for k, v in self.quan_config_seed.items()}

        for i in range(self.get_num_layers()):
            config_seed[f"model_layer_{i}"] = {
                "self_attn": {
                    "q_proj": deepcopy(self.quan_config_seed),
                    "k_proj": deepcopy(self.quan_config_seed),
                    "v_proj": deepcopy(self.quan_config_seed),
                    "out_proj": deepcopy(self.quan_config_seed),
                    "bmm_0": deepcopy(self.quan_config_seed),
                    "bmm_1": deepcopy(self.quan_config_seed),
                },
                "fc1": deepcopy(self.quan_config_seed),
                "fc2": deepcopy(self.quan_config_seed),
            }
            indexes[f"model_layer_{i}"] = {
                "self_attn": {
                    "q_proj": deepcopy(zero_seed),
                    "k_proj": deepcopy(zero_seed),
                    "v_proj": deepcopy(zero_seed),
                    "out_proj": deepcopy(zero_seed),
                    "bmm_0": deepcopy(zero_seed),
                    "bmm_1": deepcopy(zero_seed),
                },
                "fc1": deepcopy(zero_seed),
                "fc2": deepcopy(zero_seed),
            }
            index_ranges[f"model_layer_{i}"] = {
                "self_attn": {
                    "q_proj": deepcopy(ranges),
                    "k_proj": deepcopy(ranges),
                    "v_proj": deepcopy(ranges),
                    "out_proj": deepcopy(ranges),
                    "bmm_0": deepcopy(ranges),
                    "bmm_1": deepcopy(ranges),
                },
                "fc1": deepcopy(ranges),
                "fc2": deepcopy(ranges),
            }
        config_seed[f"default"] = deepcopy(self.quan_config_seed)
        indexes[f"default"] = deepcopy(zero_seed)
        index_ranges[f"default"] = deepcopy(ranges)

        config_seed["by"] = self.quan_by

        self.index_ranges = flatten_dict(index_ranges, flattened={})
        self.per_layer_search_space = {
            "self_attn": {
                "q_proj": deepcopy(ranges),
                "k_proj": deepcopy(ranges),
                "v_proj": deepcopy(ranges),
                "out_proj": deepcopy(ranges),
                "bmm_0": deepcopy(ranges),
                "bmm_1": deepcopy(ranges),
            },
            "fc1": deepcopy(ranges),
            "fc2": deepcopy(ranges),
        }
        return indexes, config_seed

    def transform_nested_dict_to_flat_dict(self, indexes):
        return flatten_dict(indexes, flattened={})

    def transform_flat_dict_to_nested_dict(self, indexes):
        return unflatten_dict(indexes)

    def get_model(self):
        return self.model["model"]

    def get_model_config(self):
        return self.get_model().config

    def rebuild_model(self, quant_config):
        model_name = opt_name_hash[self.model_name]
        if quant_config is None:
            config = self.get_model_config().from_pretrained(model_name)
        else:
            config = self.get_model_config().from_pretrained(
                model_name, quant_config=quant_config
            )
        if "cuda" in self.device.type:
            # TODO: fix this later
            # if self.model_parallel:
            #     with init_empty_weights():
            #         model = self.get_model()(config)
            #     device_map = infer_auto_device_map(
            #         model,
            #         no_split_module_classes=model._no_split_modules,
            #     )
            #     model = load_checkpoint_and_dispatch(
            #         model, checkpoint=model_name, device_map=device_map
            #     )
            # else:
            model = (
                self.get_model()
                .from_pretrained(model_name, config=config)
                .to(self.device)
            )
        elif self.device.type == "cpu":
            model = self.get_model().from_pretrained(model_name, config=config)
        else:
            raise ValueError(f"Unknown device: {self.device}")
        return model
