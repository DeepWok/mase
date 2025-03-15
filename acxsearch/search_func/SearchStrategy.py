from machop.chop.passes.graph.transforms.quantize.modify import (
    create_new_fn,
    create_new_module,
)
from machop.chop.passes.graph.transforms.quantize.hash_modules.softmax import (
    _hashsoftmax,
)
from machop.chop.passes.graph.utils import (
    get_mase_op,
    get_mase_type,
    get_node_actual_target,
)
import copy
from tqdm import tqdm
import torch
import torch.nn.functional as F
from .SearchSpace import config_pruning
from .utils import set_meta_quant_cfg, _get_similarity


class SearchStrategyBase:
    def __init__(self, n, args, kwargs, search_space_class):
        self.n = n
        if n.op == "call_method":
            self.self_obj, *self.new_args = args
        else:
            self.new_args = args
        self.new_kwargs = kwargs
        self.search_space_class = search_space_class

    def get_tensor_dict(self, op):
        if op in ("conv1d", "conv2d", "linear"):
            ori_module = get_node_actual_target(self.n)

            tensor_dict = {
                "act": self.new_args[0],
                "w": ori_module.weight,
                "b": ori_module.bias if ori_module.bias is not None else torch.rand(1),
            }
        elif op in ("add", "relu", "sub", "mul", "matmul"):
            tensor_dict = {
                "act": self.new_args[0],
                "w": self.new_args[1],
                "b": torch.rand(1),
            }
        else:
            raise NotImplementedError

        tensor_shape = [
            tensor_dict["act"].shape,
            tensor_dict["w"].shape,
            tensor_dict["b"].shape,
        ]
        return tensor_dict, tensor_shape

    def get_search_space(self, tensor_dict):
        self.search_space_class.tensor_dict = tensor_dict
        search_spaces = self.search_space_class.build_search_space(get_mase_op(self.n))
        return search_spaces

    def software_metric(self, tensor_raw, tensor_sim, metric):
        return _get_similarity(tensor_raw, tensor_sim, metric).mean()

    def hardware_metric(self, config, tensor_shape):
        def iter_bit_width(data_type, size, config, total_bit_width, total_size):
            quant_name = config.get("name")
            partial_total_size = int(torch.prod(size))
            match quant_name:
                case "integer" | "scale_integer":
                    bit_width = config.get(f"{data_type}_width")
                    total_bit_width += bit_width * partial_total_size
                    total_size += partial_total_size
                    return total_bit_width, total_size
                case "block_fp":
                    assert (
                        config.get(f"{data_type}_exponent_bias") == None
                    ), "only assume exponent_bias == None"
                    width, exponent_width, block_size = (
                        config.get(f"{data_type}_width"),
                        config.get(f"{data_type}_exponent_width"),
                        config.get(f"{data_type}_block_size"),
                    )
                    bit_width = width + exponent_width / block_size
                    total_bit_width += bit_width * partial_total_size
                    total_size += partial_total_size
                    return total_bit_width, total_size
                case _:
                    raise NotImplementedError(f"{quant_name} is not considered")

        total_bit_width = 0
        total_size = 0
        data_type = ["data_in", "weight", "bias"]
        for i, type in enumerate(data_type):
            if config.get(f"{type}_width") != None:
                total_bit_width, total_size = iter_bit_width(
                    type, tensor_shape[i], config, total_bit_width, total_size
                )
        avg_bit_width = total_bit_width / total_size

        return avg_bit_width

    def search_metric(self, ori_result, new_result, config, tensor_shape, alpha):

        software_metric = self.software_metric(ori_result, new_result, "cosine")
        hardware_metric = self.hardware_metric(config, tensor_shape)
        metric = (1000 * (1 - software_metric)) ** 2 + alpha * hardware_metric
        metric_info = {
            "software_metric": float(software_metric),
            "hardware_metric": float(hardware_metric),
        }
        return metric, metric_info

    def __call__(self, ori_result, hardware_importance):
        metrics = []
        metric_infos = []
        new_results = []

        tensor_dict, tensor_shape = self.get_tensor_dict(get_mase_op(self.n))
        search_spaces = self.get_search_space(tensor_dict)
        q = tqdm(search_spaces, desc=f"{get_mase_op(self.n)} node: {self.n.name}")
        for config in q:
            if self.n.op == "call_module":
                ori_module = get_node_actual_target(self.n)
                new_module = create_new_module(
                    get_mase_op(self.n), ori_module, config, self.n.meta
                )
                new_result = new_module(*self.new_args, **self.new_kwargs)
            elif get_mase_type(self.n) in [
                "builtin_func",
                "module_related_func",
            ]:
                new_f, args, kwargs = create_new_fn(self.n, config)
                new_result = new_f(*self.new_args, **kwargs)
            else:
                raise NotImplementedError("node should be in the quantized op")

            search_metric_kwargs = {
                "ori_result": ori_result,
                "new_result": new_result,
                # "op": get_mase_op(self.n),
                "config": config,
                "tensor_shape": [torch.tensor(i) for i in tensor_shape],
                "alpha": hardware_importance,
            }
            metric, metric_info = self.search_metric(**search_metric_kwargs)
            metrics.append(metric)
            metric_infos.append(metric_info)
            new_results.append(new_result)
        a_best_index = torch.tensor(metrics).argmin(dim=0)
        best_info, best_config = (
            metric_infos[a_best_index],
            search_spaces[a_best_index],
        )
        set_meta_quant_cfg(self.n, best_info, best_config)
        print(f"{get_mase_op(self.n)} node: {self.n.name}, similarities = {best_info}")
        return new_results[a_best_index]


class SoftmaxSearch(SearchStrategyBase):

    def hardware_metric(self, config):
        avg_bit_width = (
            config["data_in_width"]
            + config["data_in_exp_width"]
            + (config["data_in_div_frac_width"] + 1)
        ) / 3

        return avg_bit_width

    def search_metric(self, ori_result, new_result, config, alpha):

        software_metric = self.software_metric(ori_result, new_result, "cosine").mean()
        hardware_metric = self.hardware_metric(config)
        metric = (1000 * (1 - software_metric)) ** 2 + alpha * hardware_metric
        metric_info = {
            "software_metric": float(software_metric),
            "hardware_metric": float(hardware_metric),
        }
        return metric, metric_info

    def __call__(self, ori_result, hardware_importance):
        metrics = []
        metric_infos = []
        new_results = []
        q = tqdm(self.search_space_class, desc=f"{get_mase_op(self.n)} node: {self.n.name}")

        for config in q:
            if get_mase_op(self.n) in ["softmax", "hash_softmax"]:
                new_result = _hashsoftmax(
                    self.self_obj, *self.new_args, **self.new_kwargs, config=config
                )
            search_metric_kwargs = {
                "ori_result": ori_result,
                "new_result": new_result,
                "config": config,
                "alpha": hardware_importance,
            }
            metric, metric_info = self.search_metric(**search_metric_kwargs)
            metrics.append(metric)
            metric_infos.append(metric_info)
            new_results.append(new_result)
        a_best_index = torch.tensor(metrics).argmin(dim=0)
        best_info, best_config = (
            metric_infos[a_best_index],
            self.search_space_class[a_best_index],
        )
        set_meta_quant_cfg(self.n, best_info, best_config)
        print(f"{get_mase_op(self.n)} node: {self.n.name}, similarities = {best_info}")
        return new_results[a_best_index]
