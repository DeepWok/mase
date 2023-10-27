import logging
import torch
import copy
import toml
from pprint import pformat

from torch import nn
from ....passes.utils import get_mase_op, get_mase_type
from ....passes.graph.mase_graph import MaseGraph


class SearchSpaceBase:
    """
    Base class for search space.

    Subclasses must implement the following methods:
    - `_post_init_setup(self) -> None`: additional setup for the subclass instance
    - `build_search_space(self) -> None`: define the search space by setting `self.choices_flattened` and `self.choice_lengths_flattened`
    - `flattened_indexes_to_config(self, indexes: dict[str, int]) -> dict[str, dict[str, int]]`: convert indexes to config, which will be passed to `rebuild_model`
    - `rebuild_model(self, config, mode="eval") -> nn.Module | MaseGraph`: rebuild the model with the given config
    ---

    What is a search space?

    During the search, the search strategy will sample using `search_space.choice_lengths_flattened`, and passes the sampled indexes back to the search space to build a new model.
    The model building is done by `flattened_indexes_to_config` and `rebuild_model`.

    ---

    Check `machop/chop/actions/search/search_space/quantize/graph.py` for an example.
    """

    def __init__(
        self,
        model: nn.Module,
        model_info,
        config: dict,
        dummy_input,
        accelerator,
    ) -> None:
        self.model = model
        self.model_info = model_info
        self.config = config
        self.dummy_input = dummy_input
        self.accelerator = accelerator

        self.choices_flattened: dict[str, list] = {}
        self.choice_lengths_flattened: dict[str, int] = {}

        self._post_init_setup()

    def _post_init_setup(self) -> None:
        """
        Post init setup.

        This is where additional config parsing and setup should be done for the subclass instance.
        """
        raise NotImplementedError()

    def build_search_space(self) -> None:
        """
        Define the search space

        This is where `self.choices_flattened` and `self.choice_lengths_flattened` should be set.
        - `self.choices_flattened`: a flattened dict of choices, where the keys are the names of the nodes, and the values are the choices for the nodes
        - `self.choice_lengths_flattened`: a flattened dict of choice lengths, where the keys are the names of the nodes, and the values are the lengths of the choices for the nodes

        The search strategy will sample using `self.choice_lengths_flattened`, and passes the sampled indexes back to the search space to build a new model.
        """
        raise NotImplementedError()

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

    def __repr__(self) -> str:
        txt = f"{self.__class__.__name__}("
        txt += f"model={self.model.__class__.__name__}, "
        txt += f"config={pformat(self.config, depth=1)}, "
        txt += f"dummy_input={pformat(self.dummy_input, depth=1)}, "
        txt += f"accelerator={self.accelerator}, "
        txt += f")"
        return txt

    # def config_sampler(
    #     self,
    # ):
    #     raise NotImplementedError()
