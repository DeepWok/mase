import copy
import torch


def config_pruning(op, config, w, b):
    def tuning_config(config, not_used_param):
        valid_config = copy.deepcopy(config)
        for key, value in config.items():
            for i in not_used_param:
                if i in key:
                    valid_config.pop(key)
        return valid_config

    # common part
    if op in ("conv1d", "conv2d", "linear", "softmax", "hash_softmax"):
        valid_config = tuning_config(config, ["None"])
    elif op in ("relu"):
        valid_config = (
            tuning_config(config, ["weight_", "bias_"]) if w == 0 & b == 0 else None
        )
    elif op in ("matmul", "add", "sub", "mul"):
        # for the case in bfp the weight quant name will automatically neglect by quant
        valid_config = tuning_config(config, ["bias_"]) if b == 0 else None
    else:
        raise NotImplementedError
    return valid_config


class SearchSpaceBase:
    def __init__(
        self, config_choice: dict, config_list: list, quant_name: "str"
    ) -> None:
        self.config_choice = config_choice
        self.config_list = config_list
        self.quant_name = quant_name
        self.tensor_dict = None

    def build_search_space_normal(self, op):
        self.act_space, self.w_space, self.b_space = (
            self.config_choice.get("act"),
            self.config_choice.get("w"),
            self.config_choice.get("b"),
        )
        config = {"name": self.quant_name}
        search_spaces = []
        for d_config in self.act_space:
            w = 0
            for w_config in self.w_space:
                b = 0
                for b_config in self.b_space:
                    for i, config_param in enumerate(self.config_list):
                        config[f"data_in_{config_param}"] = d_config[i]
                        config[f"weight_{config_param}"] = w_config[i]
                        config[f"bias_{config_param}"] = b_config[i]
                        valid_config = config_pruning(op, config, w, b)
                    search_spaces.append(valid_config) if valid_config else None
                    b += 1
                w += 1
        return search_spaces
    
    def build_search_space(self, op):
        return self.build_search_space_normal(op)


class QuantileSearchSpace(SearchSpaceBase):
    def _quantile(self, tensor, quantile):
        if tensor.numel() >= 16777216:
            # randomly get some number
            indices = torch.randperm(tensor.numel(), dtype=torch.long)[:16777216]
            return torch.quantile(tensor.view(-1)[indices], quantile).mean()
        else:
            return torch.quantile(tensor, quantile)

    def get_quantile_config_choice(self):
        new_config_choice = {}

        for key, value in self.tensor_dict.items():
            choice = []
            for quantile_choice in self.config_choice[key]:
                width, quantile = quantile_choice[0], quantile_choice[1]
                quantile_value = self._quantile(value.abs(), quantile)
                int_width = torch.clamp(torch.ceil(quantile_value), 1, width - 1)
                new_tuple = []
                for i, item in enumerate(quantile_choice):
                    if self.config_list == ["width", "frac_width"]:
                        item = item if i != 1 else int(width - int_width)
                    else:
                        scale = float((2 ** (width - 1) - 1) / quantile_value)
                        scale = scale if scale < 10000 else 10000.0
                        item = item if i != 1 else scale

                    new_tuple.append(item)
                choice.append(tuple(new_tuple))
            choice = list(set(choice))
            new_config_choice[key] = copy.deepcopy(choice)
        self.config_choice = new_config_choice

    def build_search_space(self, op):
        self.get_quantile_config_choice()
        return self.build_search_space_normal(op)

class SoftmaxSearchSpace(SearchSpaceBase):
    def build_search_space(self, op):
        self.act_space = self.config_choice.get("act")
        config = {"name": self.quant_name}
        search_spaces = []
        for d_config in self.act_space:
            for i, config_param in enumerate(self.config_list):
                config[f"data_in_{config_param}"] = d_config[i]
                valid_config = copy.deepcopy(config)
            search_spaces.append(valid_config) if valid_config else None
        return search_spaces
