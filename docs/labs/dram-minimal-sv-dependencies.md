# Minimal New `.sv` Dependencies for DRAM-Streamed MLP (with Testbench)

## Goal
Make the DRAM-based MLP path compile and run with the existing cocotb testbench, with the minimum number of new RTL files.

## Short Answer
- For the MLP compute path itself, **0 new compute `.sv` files are required**.
- `fixed_linear.sv` already supports streamed `weight`/`bias` with valid/ready.
- DRAM mode changes who drives those ports (external stream) rather than changing linear math hardware.

## What to change in codegen (so no new compute module is needed)
1. Remove the `_dram` module renaming in `src/chop/passes/graph/transforms/verilog/emit_top.py`.
2. Keep module name as `fixed_linear`.
3. Keep DRAM behavior controlled by interface metadata (`storage="DRAM"`) and top-level parameter ports.

## Where to add DRAM-only deployment dependencies
The primary file to control per-node dependency lists is:
- `src/chop/passes/graph/analysis/add_metadata/add_hardware_metadata.py`

Use `add_component_source(...)` to switch dependency files only when DRAM mode is requested.

Simplest DRAM-only pattern:
```python
elif mase_op in INTERNAL_COMP.keys():
    node.meta["mase"]["hardware"]["toolchain"] = "INTERNAL_RTL"
    node.meta["mase"]["hardware"]["module"] = INTERNAL_COMP[mase_op][0]["name"]
    node.meta["mase"]["hardware"]["dependence_files"] = INTERNAL_COMP[mase_op][0]["dependence_files"]

    if pass_args.get("interface", {}).get("storage", "BRAM") == "DRAM":
        dram_extra = pass_args.get("interface", {}).get("dram_deployment_deps", [])
        node.meta["mase"]["hardware"]["dependence_files"] = (
            node.meta["mase"]["hardware"]["dependence_files"] + dram_extra
        )
```

Notes:
- Keep compute dependencies unchanged for simulation.
- Append only deployment adapters (AXIS formatter/scheduler) under DRAM.
- Keep `storage` explicit as `"DRAM"` for off-chip flow.

If you want this keyed by op type (for example only `linear`), the cleanest place for static lists is:
- `src/chop/passes/graph/analysis/add_metadata/hardware_metadata_layers.py`

Example structure (conceptual):
```python
DRAM_EXTRA_DEPENDENCIES = {
    "linear": [
        "memory_adapters/rtl/axis_param_reformatter.sv",
        "memory_adapters/rtl/param_stream_scheduler.sv",
    ]
}
```

Then in `add_component_source(...)`, append `DRAM_EXTRA_DEPENDENCIES.get(mase_op, [])` when `storage == "DRAM"`.

## `emit_dram` function (deployment only)
This function is only for real FPGA deployment. It is not needed for cocotb simulation.

In simulation, cocotb already quantizes and streams parameter blocks directly into DRAM-backed top-level ports.
In deployment, we need deployable parameter images that PS software and DMA can read from DDR and stream to the accelerator.

```python
def emit_dram_transform_pass(graph, pass_args={}):
    """
    Emit deployable DRAM parameter images (not BRAM ROM files) for real FPGA deployment.

    Expected outputs per DRAM-backed parameter:
    - packed tensor stream image (e.g. .bin/.hex)
    - metadata sidecar (shape, precision, parallelism, beat count, ordering)

    Notes:
    - Not required for cocotb simulation, because cocotb drivers already stream quantized parameters directly.
    - Required for deployment where PS software / DMA reads from DDR and streams to top-level parameter ports.
    """
```

Recommended `pass_args` fields:
- `project_dir`: output root
- `format`: `bin` or `hex`
- `endianness`: `little` or `big`
- `align_bytes`: beat alignment for DMA
- `emit_metadata`: `True/False`
- `target`: board/deployment tag

## Minimum deployment stack (real FPGA)
Vivado IP helps with transport, but Vivado IP alone is not enough to satisfy model-specific packing, ordering, replay, and control requirements.

1. Parameter image emitter (software/pass side)
Converts quantized tensors into stream-ready images matching hardware beat format and ordering.
2. DDR buffer allocation + loader software (PS side)
Allocates contiguous buffers, loads emitted images, and passes addresses/lengths to DMA/control plane.
3. AXI DMA (Vivado IP)
Moves parameter beats from DDR to AXI-Stream.
4. AXIS packet/beat reformatter (custom RTL/HLS)
Adapts DMA stream width/packet structure to `fc1_weight` / `fc1_bias` port beat shape.
5. Parameter stream scheduler/control (custom RTL/HLS or PS-driven FSM)
Controls when each parameter stream starts, stops, and repeats per inference, while respecting back-pressure.
6. Optional AXIS FIFO/width converter (Vivado IP)
Used when clock-domain crossing, buffering, or width adaptation is needed.

### Vivado IP vs custom logic
Vivado can provide:
- AXI DMA
- AXIS FIFO
- AXIS data width converter
- AXI interconnect / SmartConnect
- Clocking/reset and standard infrastructure

You still need to implement:
- Parameter block ordering/packing contract used by `fixed_linear`
- Tensor replay policy (e.g. re-stream full weights per sample/batch)
- Stream scheduling across weight/bias channels
- Control/status integration between PS and accelerator
- Model-specific handshaking and sequencing correctness

## Final handoff list for teammate (today)
- Do not create `fixed_linear_dram.sv`.
- Remove `_dram` renaming in `src/chop/passes/graph/transforms/verilog/emit_top.py` so top instantiates `fixed_linear`.
- Add DRAM-only deployment dependencies in `src/chop/passes/graph/analysis/add_metadata/add_hardware_metadata.py` by appending an extra list only when `storage == "DRAM"`.
- Keep deployment adapters separate from compute RTL (compute stays `fixed_linear`).
