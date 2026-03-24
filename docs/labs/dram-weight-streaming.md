#  Weight Streaming in MASE Hardware Emit Flow

## Overview

By default, MASE generates FPGA hardware that stores all layer weights and biases
**on-chip in BRAM ROM modules**. This guide explains how to instead route weights
and biases from **off-chip **, exposing them as streaming handshake ports on
the generated `top.sv` module.

The  flow is useful when:
- Weight tensors are too large to fit in on-chip BRAM.
- You want to connect to an external AXI memory controller (e.g., DDR via Zynq PS).
- You are benchmarking off-chip memory bandwidth vs on-chip compute throughput.

---

## What Changes Between BRAM and  Modes

| Aspect | BRAM (default) |  |
|---|---|---|
| Weight storage | On-chip ROM (`fc1_weight_source.sv`) | External — driven into top-level ports |
| `top.sv` interface | Only `data_in`/`data_out` ports | Also exposes `{node}_{param}`, `{node}_{param}_valid`, `{node}_{param}_ready` |
| `emit_bram` pass | Generates `.sv` ROM + `.dat` init file | Skips generation, deletes stale files |
| Testbench | BRAM source drives weights internally | Cocotb `StreamDriver` drives weight ports |

---

## How to Use

### Step 1 — Pass `storage: ` to the hardware metadata pass

```python
mg, _ = add_hardware_metadata_analysis_pass(mg, {"interface": {"storage": ""}})
```

This tags every non-input tensor parameter (weight, bias, …) in the hardware
metadata with `storage: ""` instead of the default `"BRAM"`.

To use BRAM (the original behaviour), omit the argument or pass `"BRAM"` explicitly:

```python
mg, _ = add_hardware_metadata_analysis_pass(mg, {"interface": {"storage": "BRAM"}})
# or simply:
mg, _ = add_hardware_metadata_analysis_pass(mg)
```

### Step 2 — Run the emit passes as normal

```python
mg, _ = emit_verilog_top_transform_pass(mg)   # generates top.sv with  ports
mg, _ = emit_internal_rtl_transform_pass(mg)  # copies component RTL files
mg, _ = emit_bram_transform_pass(mg)          # skips ROM generation for  params;
                                               # deletes any stale BRAM .sv/.dat files
mg, _ = emit_cocotb_transform_pass(mg)        # testbench with  StreamDrivers
```

### Step 3 — Simulate

```python
from chop.actions import simulate
simulate(skip_build=False, skip_test=False, waves=True)
```

The cocotb testbench will automatically drive `fc1_weight`, `fc1_weight_valid`,
`fc1_weight_ready` (and equivalent bias ports) from the preloaded quantised
parameter tensors.

---

## Generated `top.sv` Interface

For an MLP with a single `fc1` linear layer using  storage, `top.sv` will
expose the following additional ports:

```systemverilog
// Activation data (unchanged from BRAM flow)
input  [DATA_IN_0_PRECISION_0-1:0]  data_in_0  [DATA_IN_0_PARALLELISM_DIM_0*...-1:0],
input  data_in_0_valid,
output data_in_0_ready,

output [DATA_OUT_0_PRECISION_0-1:0] data_out_0 [DATA_OUT_0_PARALLELISM_DIM_0*...-1:0],
output data_out_0_valid,
input  data_out_0_ready,

//  weight streaming ports  <-- NEW
input  [fc1_WEIGHT_PRECISION_0-1:0] fc1_weight [fc1_WEIGHT_PARALLELISM_DIM_0*fc1_WEIGHT_PARALLELISM_DIM_1-1:0],
input  fc1_weight_valid,
output fc1_weight_ready,

//  bias streaming ports    <-- NEW
input  [fc1_BIAS_PRECISION_0-1:0]   fc1_bias   [fc1_BIAS_PARALLELISM_DIM_0*fc1_BIAS_PARALLELISM_DIM_1-1:0],
input  fc1_bias_valid,
output fc1_bias_ready,
```

Each parameter uses a **valid/ready handshake**, matching the existing streaming
interface of the internal compute modules (e.g. `fixed_linear`).

---

## Connecting to a Real Memory Controller (FPGA Deployment)

In simulation the cocotb testbench acts as the memory controller. For real FPGA
deployment you need to replace that role with an AXI-stream or custom DMA engine.
A typical integration looks like:

```
DDR ──► AXI DMA ──► AXI-Stream ──► (serialiser/reformatter) ──► fc1_weight port on top.sv
```

Key considerations:

1. **Data width**: the `fc1_weight` port is `PRECISION_0 * PARALLELISM_DIM_0 * PARALLELISM_DIM_1`
   bits wide per beat. Size your DMA burst accordingly.
2. **Ordering**: weights must arrive in the same row-major block order that
   `fixed_linear` expects — identical to how the BRAM ROM stored them.
3. **Cycling**: for batch inference `fixed_linear` reads the full weight matrix
   once per sample. Your controller must re-stream the weights for each input.
4. **Back-pressure**: the `_ready` signal indicates the compute module can accept
   a beat. Your controller must respect this or data will be dropped.

---

## Files Changed

| File | Change |
|---|---|
| `src/chop/passes/graph/analysis/add_metadata/add_hardware_metadata.py` | `add_component_source` now reads `storage` from `pass_args["interface"]` instead of assuming BRAM |
| `src/chop/passes/graph/transforms/verilog/emit_top.py` | `VerilogInterfaceEmitter` emits  top-level ports; `VerilogInternalComponentEmitter` skips BRAM source instantiation for  params |
| `src/chop/passes/graph/transforms/verilog/emit_bram.py` | `emit_bram_handshake` skips `.sv`/`.dat` generation for  params and deletes stale files from previous BRAM runs |
| `src/chop/passes/graph/transforms/verilog/emit_tb.py` | `_emit_cocotb_tb` creates `StreamDriver` instances for  ports and preloads quantised parameter blocks in `load_drivers` |

---

## Limitations / Future Work

- **Per-parameter granularity**: currently `storage` applies to all parameters
  globally. A future extension could allow e.g. weights on  but bias on BRAM
  by passing `{"interface": {"weight": {"storage": ""}, "bias": {"storage": "BRAM"}}}`.
- **Multi-layer support**: the  ports are per-node (prefixed with node name),
  so multi-layer models automatically get separate ports per layer.
- **Transposition**: the `transpose` flag in the interface metadata is preserved
  but not yet acted on in the  path — the data is assumed to arrive in the
  same layout as BRAM.
- **HLS toolchain**: `emit_bram_hls` has a symmetric stub that also needs 
  support if the HLS flow is used.
