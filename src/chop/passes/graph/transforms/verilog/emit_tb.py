import logging, torch
from pathlib import Path
from textwrap import indent

from chop.passes.graph.utils import vf, v2p, init_project
from chop.nn.quantizers import (
    integer_quantizer_for_hw,
    integer_quantizer,
)

logger = logging.getLogger(__name__)

from pathlib import Path

torch.manual_seed(0)

import cocotb
from mase_cocotb.testbench import Testbench
from mase_cocotb.interfaces.streaming import StreamDriver, StreamMonitor


import dill
import inspect


def _cap(name):
    """
    capitalize a string
    """
    return str(name).upper()


def _emit_cocotb_test(graph, pass_args={}):

    wait_time = pass_args.get("wait_time", 2)
    wait_unit = pass_args.get("wait_units", "ms")
    batch_size = pass_args.get("batch_size", 1)

    test_template = f"""
import cocotb

@cocotb.test()
async def test(dut):
    from pathlib import Path
    import dill
    from cocotb.triggers import Timer

    tb_path = Path.home() / ".mase" / "top" / "hardware" / "test" / "mase_top_tb"
    with open(tb_path / "tb_obj.dill", "rb") as f:
        tb = dill.load(f)(dut, fail_on_checks=True)

    await tb.initialize()

    in_tensors = tb.generate_inputs(batches={batch_size})
    exp_out = tb.model(*list(in_tensors.values()))

    tb.load_drivers(in_tensors)
    tb.load_monitors(exp_out)

    await tb.wait_end(timeout={wait_time}, timeout_unit="{wait_unit}")
"""

    tb_path = Path.home() / ".mase" / "top" / "hardware" / "test" / "mase_top_tb"
    tb_path.mkdir(parents=True, exist_ok=True)
    with open(tb_path / "test.py", "w") as f:
        f.write(test_template)


def _emit_cocotb_tb(graph):
    class MaseGraphTB(Testbench):
        def _dram_storage_enabled(self):
            """Return True if any argument is explicitly marked with DRAM storage."""
            observed_interfaces = []
            interface_count = 0

            for node in graph.fx_graph.nodes:
                if not hasattr(node, "meta") or "mase" not in node.meta:
                    continue

                node_name = vf(getattr(node, "name", "unknown_node"))
                node_mase = node.meta["mase"]
                # Metadata can be stored both in object-style
                # node.meta["mase"].parameters[...] and dict-style
                # node.meta["mase"]["hardware"][...]. Check both.
                node_params = getattr(node_mase, "parameters", None)
                if node_params is None and isinstance(node_mase, dict):
                    node_params = node_mase.get("parameters", {})

                hardware_from_params = (
                    node_params.get("hardware", {}) if isinstance(node_params, dict) else {}
                )
                hardware_from_dict = (
                    node_mase.get("hardware", {}) if isinstance(node_mase, dict) else {}
                )

                is_implicit = hardware_from_params.get(
                    "is_implicit", hardware_from_dict.get("is_implicit", False)
                )
                if is_implicit:
                    self._sim_log.info(
                        "DRAM detect: skip implicit node '%s'", node_name
                    )
                    continue

                interfaces = hardware_from_params.get("interface", {})
                source = "parameters.hardware.interface"
                if not interfaces:
                    interfaces = hardware_from_dict.get("interface", {})
                    source = "mase.hardware.interface"

                if not interfaces:
                    self._sim_log.info(
                        "DRAM detect: node '%s' has no interface metadata", node_name
                    )
                    continue

                for iface_name, interface_cfg in interfaces.items():
                    if not isinstance(interface_cfg, dict):
                        self._sim_log.info(
                            "DRAM detect: node '%s' iface '%s' from %s has non-dict config",
                            node_name,
                            iface_name,
                            source,
                        )
                        continue

                    storage = interface_cfg.get("storage", "<missing>")
                    interface_count += 1
                    if len(observed_interfaces) < 24:
                        observed_interfaces.append(f"{node_name}.{iface_name}:{storage}")

                    self._sim_log.info(
                        "DRAM detect: node='%s' iface='%s' storage='%s' source=%s",
                        node_name,
                        iface_name,
                        storage,
                        source,
                    )

                    if storage == "DRAM":
                        self._sim_log.info(
                            "DRAM detect: selected DRAM mode from node '%s' iface '%s'",
                            node_name,
                            iface_name,
                        )
                        return True

            if interface_count == 0:
                self._sim_log.warning(
                    "DRAM detect: no interface entries were found; defaulting to BRAM mode"
                )
            else:
                self._sim_log.warning(
                    "DRAM detect: scanned %d interface entries, no DRAM found. Sample: %s",
                    interface_count,
                    ", ".join(observed_interfaces),
                )

            return False

        def _discover_dram_param_specs_from_dut(self, dut):
            """Discover streamable parameter ports by probing DUT signal names.

            This is a robust fallback when graph interface metadata is absent in
            serialized testbench objects.
            """
            specs = {}
            for full_name, param_tensor in self.model.named_parameters():
                if "." not in full_name:
                    continue
                module_name, arg = full_name.rsplit(".", 1)
                node_name = vf(module_name)
                port_name = f"{node_name}_{arg}"
                required = [port_name, f"{port_name}_valid", f"{port_name}_ready"]
                missing = [sig for sig in required if not hasattr(dut, sig)]
                if missing:
                    continue
                specs[port_name] = (node_name, arg, param_tensor)
            return specs

        def __init__(self, dut, fail_on_checks=True):
            super().__init__(dut, dut.clk, dut.rst, fail_on_checks=fail_on_checks)
            self._sim_log = getattr(dut, "_log", logger)

            # Instantiate as many drivers as required inputs to the model
            self.input_drivers = {}
            self.output_monitors = {}

            for node in graph.nodes_in:
                for arg in node.meta["mase"]["common"]["args"].keys():
                    if "data_in" not in arg:
                        continue
                    self.input_drivers[arg] = StreamDriver(
                        dut.clk,
                        getattr(dut, arg),
                        getattr(dut, f"{arg}_valid"),
                        getattr(dut, f"{arg}_ready"),
                    )
                    self.input_drivers[arg].log.setLevel(logging.DEBUG)

            # Instantiate as many monitors as required outputs
            for node in graph.nodes_out:
                for result in node.meta["mase"]["common"]["results"].keys():
                    if "data_out" not in result:
                        continue
                    self.output_monitors[result] = StreamMonitor(
                        dut.clk,
                        getattr(dut, result),
                        getattr(dut, f"{result}_valid"),
                        getattr(dut, f"{result}_ready"),
                        check=False,
                    )
                    self.output_monitors[result].log.setLevel(logging.DEBUG)

            self.model = graph.model

            # To do: precision per input argument
            self.input_precision = graph.meta["mase"]["common"]["args"]["data_in_0"][
                "precision"
            ]

            # Create StreamDriver instances for streamed parameter ports.
            # Map model parameters to top-level DUT ports by name, e.g.:
            #   fc1.weight -> fc1_weight / fc1_weight_valid / fc1_weight_ready
            self.dram_drivers = {}
            self.dram_param_specs = {}
            metadata_dram_mode = self._dram_storage_enabled()
            discovered_specs = self._discover_dram_param_specs_from_dut(dut)

            self.dram_mode = metadata_dram_mode or bool(discovered_specs)
            if not metadata_dram_mode and discovered_specs:
                self._sim_log.warning(
                    "DRAM detect: metadata did not expose DRAM interfaces, "
                    "but discovered %d DRAM-style DUT parameter ports; enabling DRAM mode",
                    len(discovered_specs),
                )

            if self.dram_mode:
                for port_name, (node_name, arg, param_tensor) in discovered_specs.items():

                    self.dram_drivers[port_name] = StreamDriver(
                        dut.clk,
                        getattr(dut, port_name),
                        getattr(dut, f"{port_name}_valid"),
                        getattr(dut, f"{port_name}_ready"),
                    )
                    self.dram_param_specs[port_name] = (node_name, arg, param_tensor)
                    self.dram_drivers[port_name].log.setLevel(logging.DEBUG)
                    self._sim_log.info(
                        "Bound parameter StreamDriver for DUT port '%s'", port_name
                    )

                logger.debug("Discovered %d streamed parameter ports", len(self.dram_drivers))

                if self.dram_drivers:
                    self._sim_log.info(
                        "DRAM drivers enabled for ports: %s",
                        ", ".join(sorted(self.dram_drivers.keys())),
                    )
                else:
                    self._sim_log.warning(
                        "DRAM mode enabled, but no parameter stream ports were discovered"
                    )
            else:
                self._sim_log.info("BRAM mode detected; skipping DRAM parameter stream setup")

        def generate_inputs(self, batches):
            """
            Generate inputs for the model by sampling a random tensor
            for each input argument, according to its shape

            :param batches: number of batches to generate for each argument
            :type batches: int
            :return: a dictionary of input arguments and their corresponding tensors
            :rtype: Dict
            """
            # ! TO DO: iterate through graph.args instead to generalize
            inputs = {}
            for node in graph.nodes_in:
                for arg, arg_info in node.meta["mase"]["common"]["args"].items():
                    # Batch dimension always set to 1 in metadata
                    if "data_in" not in arg:
                        continue
                    # print(f"Generating data for node {node}, arg {arg}: {arg_info}")
                    inputs[f"{arg}"] = torch.rand(([batches] + arg_info["shape"][1:]))
            return inputs

        def load_drivers(self, in_tensors):
            # Preload DRAM parameter drivers with quantized parameter blocks.
            if self.dram_mode and self.dram_drivers:
                from mase_cocotb.utils import fixed_preprocess_tensor

                self._sim_log.info("Preloading DRAM parameter streams for %d ports", len(self.dram_drivers))
                total_blocks_queued = 0

                for port_name, (node_name, arg, param_tensor) in self.dram_param_specs.items():

                    arg_cap = _cap(arg)
                    parallelism_0 = self.get_parameter(
                        f"{node_name}_{arg_cap}_PARALLELISM_DIM_0"
                    )
                    parallelism_1 = self.get_parameter(
                        f"{node_name}_{arg_cap}_PARALLELISM_DIM_1"
                    )
                    width = self.get_parameter(f"{node_name}_{arg_cap}_PRECISION_0")
                    frac_width = self.get_parameter(f"{node_name}_{arg_cap}_PRECISION_1")

                    param_blocks = fixed_preprocess_tensor(
                        tensor=param_tensor,
                        q_config={
                            "width": width,
                            "frac_width": frac_width,
                        },
                        parallelism=[parallelism_1, parallelism_0],
                    )
                    block_size = parallelism_0 * parallelism_1
                    port_blocks_queued = 0
                    for block in param_blocks:
                        if len(block) < block_size:
                            block = block + [0] * (block_size - len(block))
                        self.dram_drivers[port_name].append(block)
                        port_blocks_queued += 1

                    total_blocks_queued += port_blocks_queued
                    self._sim_log.info(
                        "Queued %d DRAM blocks for port '%s' (block_size=%d)",
                        port_blocks_queued,
                        port_name,
                        block_size,
                    )

                self._sim_log.info("Queued %d DRAM parameter blocks in total", total_blocks_queued)
                assert total_blocks_queued > 0, (
                    "DRAM mode detected, but zero DRAM parameter blocks were queued. "
                    "Check DRAM interface metadata/parameter extraction."
                )
            elif self.dram_mode:
                self._sim_log.warning("DRAM mode enabled, but no DRAM parameter streams to preload")
            else:
                self._sim_log.info("No DRAM parameter streams to preload")

            self._sim_log.info("Loading input drivers for %d tensor inputs", len(in_tensors))
            for arg, arg_batches in in_tensors.items():
                # DiffLogic: do not need precision, fully unrolled so 1 batch only
                if "difflogic" in graph.nodes_in[0].meta["mase"]["hardware"]["module"]:
                    block = arg_batches[0].round().int().tolist()
                    if isinstance(block[0], list):
                        out = []
                        for row in block:
                            num = ""
                            for i in range(len(row) - 1, -1, -1):
                                num += str(row[i])
                            num = int(num, 2)
                            out.append(num)
                    else:
                        num = ""
                        for i in range(len(block) - 1, -1, -1):
                            num += str(block[i])
                        num = int(num, 2)
                        out = [num]
                    self.input_drivers[arg].append(out)
                    continue

                # Quantize input tensor according to precision
                if len(self.input_precision) > 1:
                    from mase_cocotb.utils import fixed_preprocess_tensor

                    in_data_blocks = fixed_preprocess_tensor(
                        tensor=arg_batches,
                        q_config={
                            "width": self.get_parameter(f"{_cap(arg)}_PRECISION_0"),
                            "frac_width": self.get_parameter(
                                f"{_cap(arg)}_PRECISION_1"
                            ),
                        },
                        parallelism=[
                            self.get_parameter(f"{_cap(arg)}_PARALLELISM_DIM_1"),
                            self.get_parameter(f"{_cap(arg)}_PARALLELISM_DIM_0"),
                        ],
                    )

                else:
                    # TO DO: convert to integer equivalent of floating point representation
                    pass

                # Append all input blocks to input driver
                # ! TO DO: generalize
                block_size = self.get_parameter(
                    "DATA_IN_0_PARALLELISM_DIM_0"
                ) * self.get_parameter("DATA_IN_0_PARALLELISM_DIM_1")
                for block in in_data_blocks:
                    if len(block) < block_size:
                        block = block + [0] * (block_size - len(block))
                    self.input_drivers[arg].append(block)

        def load_monitors(self, expectation):
            # DiffLogic: do not need precision, fully unrolled so 1 batch only
            if "difflogic" in graph.nodes_out[0].meta["mase"]["hardware"]["module"]:
                self.output_monitors["data_out_0"].expect(expectation)
                self.output_monitors["data_out_0"].in_flight = True
                return

            from mase_cocotb.utils import fixed_preprocess_tensor

            # Process the expectation tensor
            output_blocks = fixed_preprocess_tensor(
                tensor=expectation,
                q_config={
                    "width": self.get_parameter(f"DATA_OUT_0_PRECISION_0"),
                    "frac_width": self.get_parameter(f"DATA_OUT_0_PRECISION_1"),
                },
                parallelism=[
                    self.get_parameter(f"DATA_OUT_0_PARALLELISM_DIM_1"),
                    self.get_parameter(f"DATA_OUT_0_PARALLELISM_DIM_0"),
                ],
            )

            # Set expectation for each monitor
            for block in output_blocks:
                # ! TO DO: generalize to multi-output models
                if len(block) < self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0"):
                    block = block + [0] * (
                        self.get_parameter("DATA_OUT_0_PARALLELISM_DIM_0") - len(block)
                    )
                self.output_monitors["data_out_0"].expect(block)

            # Drive the in-flight flag for each monitor
            self.output_monitors["data_out_0"].in_flight = True

    # Serialize testbench object to be instantiated within test by cocotb runner
    cls_obj = MaseGraphTB
    tb_path = Path.home() / ".mase" / "top" / "hardware" / "test" / "mase_top_tb"
    tb_path.mkdir(parents=True, exist_ok=True)
    with open(tb_path / "tb_obj.dill", "wb") as file:
        dill.dump(cls_obj, file)
    with open(tb_path / "__init__.py", "w") as file:
        file.write("from .test import test")


def emit_cocotb_transform_pass(graph, pass_args={}):
    """
    Emit test bench and related files for simulation

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: this pass requires additional arguments which is explained below, defaults to {}
    :type pass_args: _type_, optional
    :return: return a tuple of a MaseGraph and an empty dict (no additional info to return)
    :rtype: tuple(MaseGraph, Dict)

    - pass_args
        - project_dir -> str : the directory of the project
        - trace -> bool : trace waves in the simulation
    """
    logger.info("Emitting testbench...")
    project_dir = (
        pass_args["project_dir"]
        if "project_dir" in pass_args.keys()
        else Path.home() / ".mase" / "top"
    )

    init_project(project_dir)

    _emit_cocotb_test(graph, pass_args=pass_args)
    _emit_cocotb_tb(graph)

    return graph, None
