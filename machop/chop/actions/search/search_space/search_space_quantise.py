from .base import SearchSpaceBase
from chop.passes.transforms.quantize import QUANTIZEABLE_OP, quantize_transform_pass
from collections import defaultdict


class MixedPrecisionSearchSpace(SearchSpaceBase):
    default_config = {
        "name": "integer",
        "bias_frac_width": 5,
        "bias_width": 8,
        "data_in_frac_width": 5,
        "data_in_width": 8,
        "weight_frac_width": 3,
        "weight_width": 8,
    }

    def flatten_dict(self, xs):
        ys = {}
        for k, v in xs.items():
            for k2, v2 in v.items():
                ys[f"{k}.{k2}"] = v2
        return ys

    def build_search_space(self):
        self.choices = {}
        self.search_space = {}

        quantize_config = dict(self.config)
        self.name = quantize_config.pop("name")
        self.style = quantize_config.pop("style")
        for n, v in self.graph_info.items():
            if v["mase_op"] in QUANTIZEABLE_OP:
                self.choices[n] = {**quantize_config}
                self.search_space[n] = {n: len(v) for n, v in quantize_config.items()}

        self.choices_flattened = self.flatten_dict(self.choices)
        self.search_space_flattened = self.flatten_dict(self.search_space)
        self.per_layer_search_space = {n: len(v) for n, v in quantize_config.items()}
        self.num_layers = len(self.choices)

    def build_sample(self, xs, split_string="."):
        ys = defaultdict(dict)
        ys_idx = defaultdict(dict)
        for k, v in xs.items():
            k1, k2 = k.split(split_string)
            if "config" not in ys[k1]:
                ys[k1]["config"] = {"name": self.name}
            ys_idx[k1][k2] = v
            ys[k1]["config"][k2] = self.choices[k1][k2][v]
        # ys are the selected choices, ys_idx is its index
        return ys, ys_idx

    def get_model_mg(self, config):
        config["by"] = "name"
        return quantize_transform_pass(self.mg, config)

    def get_model_module(self, config):
        breakpoint()

    def get_model(self, config):
        if self.use_mg:
            return self.get_model_mg(config)
        return self.get_model_module(config)
