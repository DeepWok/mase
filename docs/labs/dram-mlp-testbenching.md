# DRAM-Backed MLP Testbenching in MASE

## 1. Scope and objective

This note documents how the MLP hardware testbench was adapted to validate off-chip parameter streaming, where weights and biases are supplied as runtime input streams instead of BRAM-backed parameter source modules.

The focus is on additions in MASE required to support this verification flow.

## 2. Verification target

For the toy MLP:

```python
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        return torch.nn.functional.relu(self.fc1(x))
```

the hardware test should:

- drive a valid activation stream into `data_in_0`
- drive valid parameter streams into `fc1_weight` and `fc1_bias`
- compare DUT output stream against software model expectation
- terminate quickly without waiting for timeout

## 3. Off-chip metadata selection

The DRAM mode is selected during hardware metadata generation:

```python
mg, _ = add_hardware_metadata_analysis_pass(mg, {"interface": {"storage": "DRAM"}})
```

In `add_component_source`, MASE selects DRAM-aware module/dependency mapping and marks tensor parameters as DRAM-backed interfaces:

```python
storage_type = pass_args.get("interface", {}).get("storage", "BRAM")
if storage_type == "DRAM":
    node.meta["mase"]["hardware"]["module"] = DRAM_INTERNAL_COMP[mase_op][0]["name"]
    node.meta["mase"]["hardware"]["dependence_files"] = DRAM_INTERNAL_COMP[mase_op][0]["dependence_files"]
...
node.meta["mase"]["hardware"]["interface"][arg] = {
    "storage": storage_type,
    "transpose": False,
}
```

Source: `src/chop/passes/graph/analysis/add_metadata/add_hardware_metadata.py`

## 4. Top-level RTL interface changes for DRAM parameters

The top emitter now exposes parameter streaming ports when storage is DRAM:

```python
if hardware.get("interface", {}).get(arg, {}).get("storage", "BRAM") != "DRAM":
    continue
...
interface += f"""
input  [{node_name}_{arg_name}_PRECISION_0-1:0] {node_name}_{arg} [{'*'.join(parallelism_params)}-1:0],
input  {node_name}_{arg}_valid,
output {node_name}_{arg}_ready,"""
```

This creates explicit DUT ports such as:

- `fc1_weight`, `fc1_weight_valid`, `fc1_weight_ready`
- `fc1_bias`, `fc1_bias_valid`, `fc1_bias_ready`

Source: `src/chop/passes/graph/transforms/verilog/emit_top.py`

## 5. Core cocotb test flow

The generated test computes expected output directly from the software model and then drives/monitors streams:

```python
in_tensors = tb.generate_inputs(batches=1)
exp_out = tb.model(*list(in_tensors.values()))

tb.load_drivers(in_tensors)
tb.load_monitors(exp_out)

await tb.wait_end(timeout=2, timeout_unit="ms")
```

Source: generated `~/.mase/top/hardware/test/mase_top_tb/test.py` from `emit_cocotb_transform_pass`.

### Expected-value definition

For a single layer MLP with ReLU:

$$
y = \mathrm{ReLU}(xW^T + b)
$$

`exp_out` is computed by the PyTorch model and then quantized/packed into output beats by `fixed_preprocess_tensor` in `load_monitors`.

## 6. MASE additions in testbench emitter for off-chip parameter driving

### 6.1 Why an additional change was needed

During debugging, parameter ports were present in generated RTL, but some serialized graph paths had incomplete FX-node metadata at cocotb runtime, causing DRAM discovery to miss all parameter streams.

### 6.2 Robust parameter-port discovery

The testbench now binds parameter drivers from `model.named_parameters()` and DUT port existence, instead of relying only on FX-node metadata traversal:

```python
self.dram_drivers = {}
self.dram_param_specs = {}
for full_name, param_tensor in self.model.named_parameters():
    if "." not in full_name:
        continue
    module_name, arg = full_name.rsplit(".", 1)
    node_name = vf(module_name)
    port_name = f"{node_name}_{arg}"
    missing = [
        sig
        for sig in [port_name, f"{port_name}_valid", f"{port_name}_ready"]
        if not hasattr(dut, sig)
    ]
    if missing:
        continue

    self.dram_drivers[port_name] = StreamDriver(
        dut.clk,
        getattr(dut, port_name),
        getattr(dut, f"{port_name}_valid"),
        getattr(dut, f"{port_name}_ready"),
    )
    self.dram_param_specs[port_name] = (node_name, arg, param_tensor)
```

Source: `src/chop/passes/graph/transforms/verilog/emit_tb.py`

### 6.3 Parameter tensor packing and streaming

Each parameter tensor is quantized and packed according to emitted per-port precision and parallelism:

```python
for port_name, (node_name, arg, param_tensor) in self.dram_param_specs.items():
    arg_cap = _cap(arg)
    parallelism_0 = self.get_parameter(f"{node_name}_{arg_cap}_PARALLELISM_DIM_0")
    parallelism_1 = self.get_parameter(f"{node_name}_{arg_cap}_PARALLELISM_DIM_1")
    width = self.get_parameter(f"{node_name}_{arg_cap}_PRECISION_0")
    frac_width = self.get_parameter(f"{node_name}_{arg_cap}_PRECISION_1")

    param_blocks = fixed_preprocess_tensor(
        tensor=param_tensor,
        q_config={"width": width, "frac_width": frac_width},
        parallelism=[parallelism_1, parallelism_0],
    )

    block_size = parallelism_0 * parallelism_1
    for block in param_blocks:
        if len(block) < block_size:
            block = block + [0] * (block_size - len(block))
        self.dram_drivers[port_name].append(block)
```

This makes parameter traffic follow the same valid/ready handshake pattern as regular input streams.

Source: `src/chop/passes/graph/transforms/verilog/emit_tb.py`

## 7. Input driving and expected output monitor path

Input stream driving remains unchanged in concept: random input tensors are quantized and block-packed using interface parameters, then appended to input drivers.

Output monitoring uses quantized/packed expected blocks and checks observed beats in-order.

Representative monitor setup:

```python
self.output_monitors[result] = StreamMonitor(
    dut.clk,
    getattr(dut, result),
    getattr(dut, f"{result}_valid"),
    getattr(dut, f"{result}_ready"),
    check=False,
)
```

Source: `src/chop/passes/graph/transforms/verilog/emit_tb.py`

## 8. End-to-end lab/notebook sequence

Minimal flow used in the lab notebook:

```python
mg, _ = add_hardware_metadata_analysis_pass(mg, {"interface": {"storage": "DRAM"}})
mg, _ = emit_verilog_top_transform_pass(mg)
mg, _ = emit_internal_rtl_transform_pass(mg)
mg, _ = emit_bram_transform_pass(mg)
mg, _ = emit_cocotb_transform_pass(mg)

simulate(skip_build=False, skip_test=False, waves=True)
```

Notebook source: `docs/labs/lab4-hardware.ipynb`

## 9. Observed runtime behavior after additions

After the off-chip testbench additions:

- parameter stream drivers are bound for `fc1_weight` and `fc1_bias`
- DRAM parameter blocks are queued before main input driving
- simulation terminates in a short runtime instead of timing out

This validates that the testbench can inject a valid test case into both activation and off-chip parameter channels.

## 10. Suggested report framing (for 4-page writeup)

A clear structure for the report section is:

1. Problem statement: BRAM-only parameter sourcing limits off-chip verification.
2. Method: expose DRAM parameter ports, generate cocotb drivers for those ports, compute expected output from software model, compare stream outputs.
3. Implementation details: metadata switch, top-level port generation, robust parameter-port binding, quantized block packing.
4. Results: successful parameter+input driving and completion without timeout.
5. Limitations and future work: stricter output checks (`check=True`), multi-layer scaling, AXI adapter-level traffic modeling.
