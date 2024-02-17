import logging

from torch import nn

from ...passes.graph.utils import get_module_by_name

logger = logging.getLogger(__name__)


class MaseMetadata:
    """
    The metadata of a Mase node in a Mase graph describes the constraints of the
    node for any static analysis or possible transformation. The metadata has a
    tree structure, e.g.
    - common
      - mase_op -> str : the mase op of the node, e.g. placeholder, linear, relu
      - mase_type -> str : the mase type of the node, e.g. module, builtin_func, module_related_func
      - args -> {}
         - $name : name of the arg
           (if the arg is a tensor)
           - type -> type of the arg, e.g. fixed point or float
           - precision -> format of the type, e.g. (10, 5)
           - shape -> shape of the arg
           (if the arg is not a tensor)
           - value of the arg
      - results -> {}
         - $name : name of the result
           (if the result is a tensor)
           - type -> type of the result, e.g. fixed point or float
           - precision -> format of the type, e.g. (10, 5)
           - size -> size of the result
           (if the result is not a tensor)
           - value of the result
    - software: dict
      - args: dict
        - $name (dict): name of the arg, e.g. data_in_0
          - "stat": {"record": {"data": ..., "count": ...},
                     "variance_online": {"variance": ..., "mean": ..., "count": ...}},
                     "variance_precise": {"variance": ..., "mean": ..., "count": ...},
                     "range_n_sigma": {"min": ..., "max": ..., "count": ...},
                     "range_quantile": {"min": ..., "max": ..., "count": ...},
                     "range_min_max": {"min": ..., "max": ..., "count": ...},
                    }.
      - results: dict
        - $name (dict): name of the result, e.g. data_out_0
          - "stat": {"stat_name": { # stat_values } }
    - hardware
      - is_implicit -> bool : whether the node is mapped on hardware or software annotation only
      - verilog_param -> {} : parameters need for customise the hardware module
      - device_id -> int : the ID of the device where the node is mapped, default = -1
      - toolchain -> str : tool chain for code generation, must be INTERNAL, EXTERNAL or HLS
      - module -> str : the name of the used hardware module
      - interface -> {}
         - name : name of the parameters
           - storage : the hardware interface implemented, must be BRAM
           - transpose : whether the data needs to be transposed before emitting
      - dependence_files -> [] : the dependent files for the generated module
    ...
    """

    # Hardware dict
    known_types = ["fixed", "float", "NA"]
    known_toolchain = ["INTERNAL", "EXTERNAL", "HLS"]
    known_storage = ["BRAM"]

    def __init__(
        self,
        node=None,
        model=None,
    ):
        # Top-level model
        self.model = model
        # The fx node of the module in the fx graph of the model
        self.node = node
        # layers that we have in RTL
        self.internal_layers = {nn.Linear: "linear", nn.ReLU: "relu"}

        self.parameters = {
            "common": {},
            "software": {},
            "hardware": {},
        }

    @property
    def module(self):
        # The target module in the model
        # if it is not a "call_module" node, return None
        if self.node.op == "call_module":
            return get_module_by_name(self.model, self.node.target)
        else:
            return None

    @property
    def graph(self):
        # The fx graph of the model
        return self.model.graph

    def __getitem__(self, key):
        return self.parameters[key]

    def __setitem__(self, key, value):
        self.parameters[key] = value
