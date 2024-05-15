import torch.nn as nn
from copy import deepcopy

from ....ir.graph.mase_graph import MaseGraph
from .base import SearchSpaceBase

from ....passes.graph import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
)
from .utils import flatten_dict, unflatten_dict


class SystolicMappingSearchSpace(SearchSpaceBase):
    """
    Search space for systolic array mapping.
    """

    def _post_init_setup(self) -> None:
        """
        Post init setup.

        This is where additional config parsing and setup should be done for the subclass instance.
        """
        pass

    def build_search_space(self) -> None:
        """
        Define the search space

        This is where `self.choices_flattened` and `self.choice_lengths_flattened` should be set.
        - `self.choices_flattened`: a flattened dict of choices, where the keys are the names of the nodes, and the values are the choices for the nodes
        - `self.choice_lengths_flattened`: a flattened dict of choice lengths, where the keys are the names of the nodes, and the values are the lengths of the choices for the nodes

        The search strategy will sample using `self.choice_lengths_flattened`, and passes the sampled indexes back to the search space to build a new model.
        """
        # Create masegraph
        assert self.model_info.is_fx_traceable, "Model must be fx traceable"
        mg = MaseGraph(self.model)
        mg, _ = init_metadata_analysis_pass(mg, None)
        mg, _ = add_common_metadata_analysis_pass(
            mg, {"dummy_in": self.dummy_input, "force_device_meta": False}
        )
        mg, _ = add_hardware_metadata_analysis_pass(mg, None)
        self.mg = mg

        # Build the search space
        choices = {}
        seed = self.config["seed"]
        for node in [
            node
            for node in self.mg.nodes
            if not node.meta["mase"]["hardware"]["is_implicit"]
        ]:
            op = node.meta["mase"]["common"]["mase_op"]
            if op in seed.keys():
                choices[node.name] = deepcopy(seed[op])
            else:
                choices[node.name] = deepcopy(seed["default"])

        # flatten the choices and choice_lengths
        flatten_dict(choices, flattened=self.choices_flattened)
        self.choice_lengths_flattened = {
            k: len(v) for k, v in self.choices_flattened.items()
        }

    def flattened_indexes_to_config(
        self, indexes: dict[str, int]
    ) -> dict[str, dict[str, int]]:
        """
        Convert indexes to config which will be passed to `rebuild_model`.

        It is recommended to make the returned config ready-to-use,
        because later the search strategy can easily convert logged sampled config to ready-to-use config dict using this method.

        """
        raise NotImplementedError()

    def rebuild_model(
        self, sampled_config, is_eval_mode: bool
    ) -> nn.Module | MaseGraph:
        """
        Rebuild the model with the given config
        """
        raise NotImplementedError()
