# This is the search space for mixed-precision post-training-quantization quantization search on mase graph.
from copy import deepcopy
from torch import nn
from ..base import SearchSpaceBase
from .....passes.graph.transforms.quantize import (
    QUANTIZEABLE_OP,
    quantize_transform_pass,
)
from .....ir.graph.mase_graph import MaseGraph
from .....passes.graph import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
)
from .....passes.graph.utils import get_mase_op, get_mase_type
from ..utils import flatten_dict, unflatten_dict
from collections import defaultdict

DEFAULT_QUANTIZATION_CONFIG = {
    "config": {
        "name": "integer",
        "bypass": True,
        "bias_frac_width": 5,
        "bias_width": 8,
        "data_in_frac_width": 5,
        "data_in_width": 8,
        "weight_frac_width": 3,
        "weight_width": 8,
    }
}


class GraphSearchSpaceMixedPrecisionPTQ(SearchSpaceBase):
    """
    Post-Training quantization search space for mase graph.
    """

    def _post_init_setup(self):
        self.model.to("cpu")  # save this copy of the model to cpu
        self.mg = None
        self._node_info = None
        self.default_config = DEFAULT_QUANTIZATION_CONFIG

        # quantize the model by type or name
        assert (
            "by" in self.config["setup"]
        ), "Must specify entry `by` (config['setup']['by] = 'name' or 'type')"

    def rebuild_model(self, sampled_config, is_eval_mode: bool = True):
        # set train/eval mode before creating mase graph
        if is_eval_mode:
            self.model.eval()
        else:
            self.model.train()

        if self.mg is None:
            assert self.model_info.is_fx_traceable, "Model must be fx traceable"
            mg = MaseGraph(self.model)
            mg = init_metadata_analysis_pass(mg, None)
            mg = add_common_metadata_analysis_pass(mg, self.dummy_input)
            self.mg = mg
        if sampled_config is not None:
            mg = quantize_transform_pass(self.mg, sampled_config)
        mg.model.to(self.accelerator)
        return mg

    def _build_node_info(self):
        """
        Build a mapping from node name to mase_type and mase_op.
        """

    def build_search_space(self):
        """
        Build the search space for the mase graph (only quantizeable ops)
        """
        # Build a mapping from node name to mase_type and mase_op.
        mase_graph = self.rebuild_model(sampled_config=None, is_eval_mode=True)
        node_info = {}
        for node in mase_graph.fx_graph.nodes:
            node_info[node.name] = {
                "mase_type": get_mase_type(node),
                "mase_op": get_mase_op(node),
            }

        # Build the search space
        choices = {}
        seed = self.config["seed"]

        match self.config["setup"]["by"]:
            case "name":
                # iterate through all the quantizeable nodes in the graph
                # if the node_name is in the seed, use the node seed search space
                # else use the default search space for the node
                for n_name, n_info in node_info.items():
                    if n_info["mase_op"] in QUANTIZEABLE_OP:
                        if n_name in seed:
                            choices[n_name] = deepcopy(seed[n_name])
                        else:
                            choices[n_name] = deepcopy(seed["default"])
            case "type":
                # iterate through all the quantizeable nodes in the graph
                # if the node mase_op is in the seed, use the node seed search space
                # else use the default search space for the node
                for n_name, n_info in node_info.items():
                    n_op = n_info["mase_op"]
                    if n_op in QUANTIZEABLE_OP:
                        if n_op in seed:
                            choices[n_name] = deepcopy(seed[n_op])
                        else:
                            choices[n_name] = deepcopy(seed["default"])
            case _:
                raise ValueError(
                    f"Unknown quantization by: {self.config['setup']['by']}"
                )

        # flatten the choices and choice_lengths
        flatten_dict(choices, flattened=self.choices_flattened)
        self.choice_lengths_flattened = {
            k: len(v) for k, v in self.choices_flattened.items()
        }

    def flattened_indexes_to_config(self, indexes: dict[str, int]):
        """
        Convert sampled flattened indexes to a nested config which will be passed to `rebuild_model`.

        ---
        For example:
        ```python
        >>> indexes = {
            "conv1/config/name": 0,
            "conv1/config/bias_frac_width": 1,
            "conv1/config/bias_width": 3,
            ...
        }
        >>> choices_flattened = {
            "conv1/config/name": ["integer", ],
            "conv1/config/bias_frac_width": [5, 6, 7, 8],
            "conv1/config/bias_width": [3, 4, 5, 6, 7, 8],
            ...
        }
        >>> flattened_indexes_to_config(indexes)
        {
            "conv1": {
                "config": {
                    "name": "integer",
                    "bias_frac_width": 6,
                    "bias_width": 6,
                    ...
                }
            }
        }
        """
        flattened_config = {}
        for k, v in indexes.items():
            flattened_config[k] = self.choices_flattened[k][v]

        config = unflatten_dict(flattened_config)
        config["default"] = self.default_config
        config["by"] = self.config["setup"]["by"]
        return config
