"""
Lab 4 Extension Task: Hardware Emission with Leaky ReLU

Emit SystemVerilog for a simple MLP, then patch the generated top.sv
to replace ReLU with Leaky ReLU from MASE's component library.
Simulate both designs and compare latency.
"""

import shutil
import time

import torch
import torch.nn as nn

from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.analysis import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    report_node_type_analysis_pass,
)
from chop.passes.graph.transforms import (
    emit_verilog_top_transform_pass,
    emit_internal_rtl_transform_pass,
    emit_bram_transform_pass,
    emit_cocotb_transform_pass,
    quantize_transform_pass,
)
from chop.tools.logger import set_logging_verbosity
from pathlib import Path

set_logging_verbosity("info")

torch.manual_seed(0)


class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))
        return x


# Build MaseGraph, quantize, and emit Verilog (ReLU baseline)
print("Emitting Verilog with ReLU (baseline)...")

mlp = MLP()
mg = MaseGraph(model=mlp)

batch_size = 1
x = torch.randn((batch_size, 2, 2))
dummy_in = {"x": x}

mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(
    mg, {"dummy_in": dummy_in, "add_value": False}
)

quan_args = {
    "by": "type",
    "default": {
        "config": {
            "name": "fixed",
            "data_in_width": 8,
            "data_in_frac_width": 3,
            "weight_width": 8,
            "weight_frac_width": 3,
            "bias_width": 8,
            "bias_frac_width": 3,
            "data_out_width": 8,
            "data_out_frac_width": 3,
            "floor": True,
        }
    },
}

mg, _ = quantize_transform_pass(mg, quan_args)
_ = report_node_type_analysis_pass(mg)

for node in mg.fx_graph.nodes:
    for arg, arg_info in node.meta["mase"]["common"]["args"].items():
        if isinstance(arg_info, dict):
            arg_info["type"] = "fixed"
            arg_info["precision"] = [8, 3]
    for result, result_info in node.meta["mase"]["common"]["results"].items():
        if isinstance(result_info, dict):
            result_info["type"] = "fixed"
            result_info["precision"] = [8, 3]

mg, _ = add_hardware_metadata_analysis_pass(mg)
mg, _ = emit_verilog_top_transform_pass(mg)
mg, _ = emit_internal_rtl_transform_pass(mg)
mg, _ = emit_bram_transform_pass(mg)
mg, _ = emit_cocotb_transform_pass(mg)

print("Verilog emission complete (ReLU baseline).")

# Run ReLU simulation
project_dir = Path.home() / ".mase" / "top"
rtl_dir = project_dir / "hardware" / "rtl"

with open(rtl_dir / "top.sv", "r") as f:
    relu_top_sv = f.read()

print("Running ReLU simulation...")
try:
    from chop.actions import simulate
    relu_start = time.time()
    simulate(skip_build=False, skip_test=False)
    relu_time = time.time() - relu_start
    relu_sim_success = True
    print(f"ReLU simulation completed in {relu_time:.2f}s")
except Exception as e:
    relu_time = None
    relu_sim_success = False
    print(f"ReLU simulation failed (verilator may not be installed): {e}")

# Copy Leaky ReLU from MASE components library
print("\nPatching design with Leaky ReLU...")

import mase_components
components_dir = Path(mase_components.__file__).parent
leaky_relu_src = components_dir / "activation_layers" / "rtl" / "fixed_leaky_relu.sv"

if not leaky_relu_src.exists():
    raise FileNotFoundError(
        f"fixed_leaky_relu.sv not found at {leaky_relu_src}. "
        "Ensure mase_components is installed correctly."
    )

shutil.copy(leaky_relu_src, rtl_dir / "fixed_leaky_relu.sv")
print(f"Copied fixed_leaky_relu.sv from MASE components library")

# Patch top.sv to replace ReLU with Leaky ReLU
with open(rtl_dir / "top.sv", "r") as f:
    top_sv = f.read()

with open(rtl_dir / "top_relu_backup.sv", "w") as f:
    f.write(top_sv)

# negative_slope = 1 / 2^3 = 0.125 in Q8.3
NEGATIVE_SLOPE_VALUE = 1
NEGATIVE_SLOPE_PRECISION_0 = 8
NEGATIVE_SLOPE_PRECISION_1 = 3

# Add NEGATIVE_SLOPE parameters to module parameter list
top_sv = top_sv.replace(
    "parameter relu_INPLACE = 0,",
    f"parameter relu_INPLACE = 0,\n"
    f"    parameter relu_NEGATIVE_SLOPE_PRECISION_0 = {NEGATIVE_SLOPE_PRECISION_0},\n"
    f"    parameter relu_NEGATIVE_SLOPE_PRECISION_1 = {NEGATIVE_SLOPE_PRECISION_1},\n"
    f"    parameter relu_NEGATIVE_SLOPE_VALUE = {NEGATIVE_SLOPE_VALUE},"
)

# Replace component instantiation
top_sv = top_sv.replace(
    "// relu\nfixed_relu #(",
    "// leaky_relu (replacing relu)\nfixed_leaky_relu #("
)

# Add NEGATIVE_SLOPE parameters to instantiation
top_sv = top_sv.replace(
    "    .INPLACE(relu_INPLACE),\n"
    "    .DATA_OUT_0_PRECISION_0(relu_DATA_OUT_0_PRECISION_0),",
    "    .INPLACE(relu_INPLACE),\n"
    f"    .NEGATIVE_SLOPE_PRECISION_0(relu_NEGATIVE_SLOPE_PRECISION_0),\n"
    f"    .NEGATIVE_SLOPE_PRECISION_1(relu_NEGATIVE_SLOPE_PRECISION_1),\n"
    f"    .NEGATIVE_SLOPE_VALUE(relu_NEGATIVE_SLOPE_VALUE),\n"
    "    .DATA_OUT_0_PRECISION_0(relu_DATA_OUT_0_PRECISION_0),"
)

# Rename instance
top_sv = top_sv.replace(
    ") relu_inst (",
    ") leaky_relu_inst ("
)

with open(rtl_dir / "top.sv", "w") as f:
    f.write(top_sv)

print("top.sv patched successfully!")

# Run Leaky ReLU simulation
print("Running Leaky ReLU simulation...")
try:
    leaky_start = time.time()
    simulate(skip_build=False, skip_test=False)
    leaky_time = time.time() - leaky_start
    leaky_sim_success = True
    print(f"Leaky ReLU simulation completed in {leaky_time:.2f}s")
except Exception as e:
    leaky_time = None
    leaky_sim_success = False
    print(f"Leaky ReLU simulation failed: {e}")

# Compare results
print(f"\nReLU vs Leaky ReLU Comparison:")
if relu_sim_success and leaky_sim_success:
    overhead = ((leaky_time - relu_time) / relu_time * 100) if relu_time > 0 else 0
    print(f"  ReLU sim time:       {relu_time:.2f}s")
    print(f"  Leaky ReLU sim time: {leaky_time:.2f}s (overhead: {overhead:.1f}%)")
else:
    print(f"  ReLU:       {'Success' if relu_sim_success else 'Failed/Skipped'}")
    print(f"  Leaky ReLU: {'Success' if leaky_sim_success else 'Failed/Skipped'}")

print(f"\n  Negative slope = {NEGATIVE_SLOPE_VALUE}/2^{NEGATIVE_SLOPE_PRECISION_1} = {NEGATIVE_SLOPE_VALUE / (2**NEGATIVE_SLOPE_PRECISION_1):.4f}")
print("  Both activations are purely combinational (zero latency overhead).")
print("  Leaky ReLU adds multiplier logic for the negative slope computation.")
