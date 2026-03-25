# Initial Cocotb Plan: DRAM-Style MLP Testbench Only

## Objective
Deliver working testbenches for off-chip weight streaming on a small MLP.
This plan is testbench-focused only and does not include real FPGA integration work.

## Scope
In scope:
- Generate DRAM-style `top.sv` for an MLP
- Remove BRAM parameter source usage
- Drive `weight` and `bias` from cocotb (DRAM emulation)
- Verify functional correctness and basic throughput behavior in simulation

Out of scope:
- AXI DMA integration
- Vivado block design and deployment adapters
- Board bring-up

## 3-Step Execution Plan

1. Take a normal generated MLP and regenerate in DRAM mode
- Start from the existing MLP emit flow.
- Use hardware metadata storage `"DRAM"` so parameters are off-chip streamed.
- Confirm `top.sv` is generated with top-level parameter ports:
  - `{node}_{param}`
  - `{node}_{param}_valid`
  - `{node}_{param}_ready`

2. Ensure `top.sv` has no BRAM-initialized parameter sources
- Verify there are no `*_source.sv` parameter module instantiations for weight/bias.
- Verify `emit_bram` skipped DRAM parameter ROM generation.
- Keep compute logic unchanged (`fixed_linear` stays the math core).

3. Write cocotb to emulate DRAM and verify behavior
- In testbench, drive both activations and parameters.
- For parameters (`weight`, `bias`):
  - quantize with the same precision as hardware params
  - pack into blocks by parallelism
  - stream with valid/ready handshake
- Verify:
  - output correctness vs software model
  - no handshake drops/deadlocks
  - basic simulation-level performance counters (cycles/latency per sample)

## Minimal Touch Points
- `src/chop/passes/graph/transforms/verilog/emit_top.py`
- `src/chop/passes/graph/transforms/verilog/emit_bram.py`
- `src/chop/passes/graph/transforms/verilog/emit_tb.py`
- One DRAM-mode test under `test/passes/graph/transforms/verilog/`

## Acceptance Criteria
1. DRAM-mode `top.sv` compiles and has exposed parameter streaming ports.
2. No BRAM parameter source modules are instantiated for DRAM params.
3. Cocotb test passes for dummy MLP with DRAM-emulated `weight` and `bias` streams.
4. Collected basic performance metrics in simulation (for example cycle count to first output and total inference cycles).