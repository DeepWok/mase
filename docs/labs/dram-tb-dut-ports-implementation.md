# DRAM TB DUT Ports: Short Plan

## Scope
- DRAM mode only: expose parameter stream ports.
- BRAM mode: no extra parameter ports.

## Required DRAM Ports
For each DRAM parameter:
- `<node>_<param>`
- `<node>_<param>_valid`
- `<node>_<param>_ready`

MLP example:
- `fc1_weight`, `fc1_weight_valid`, `fc1_weight_ready`
- `fc1_bias`, `fc1_bias_valid`, `fc1_bias_ready`

## How original input is handled (reference)
`data_in*` flow is:
1. Create `StreamDriver(dut.<arg>, <arg>_valid, <arg>_ready)`.
2. Generate tensors from metadata shape.
3. Quantize+pack with `fixed_preprocess_tensor(...)`.
4. Append blocks to driver.

DRAM parameters should use the same pattern.

## Verification plan
1. Runtime TB check only
- Discover expected DRAM ports from metadata (`storage == DRAM`).
- At TB init, log expected port count.
- For each expected port, verify DUT has all three signals:
	- `<node>_<param>`
	- `<node>_<param>_valid`
	- `<node>_<param>_ready`
- Bind a `StreamDriver` and log the bound port name.
- Assert: if DRAM metadata exists, bound driver count must equal expected count.

2. Runtime preload check
- During `load_drivers`, log queued block count per DRAM port.
- Log total queued DRAM blocks.
- Assert total queued DRAM blocks > 0 in DRAM mode.

## Driving plan
For each DRAM port in TB:
1. Bind `StreamDriver` to `<node>_<param>`, valid, ready.
2. Read parameter tensor from model (`module.get_parameter(arg)`).
3. Read DUT metadata parameters for that port:
	- precision (`*_PRECISION_0`, `*_PRECISION_1`)
	- parallelism (`*_PARALLELISM_DIM_0`, `*_PARALLELISM_DIM_1`)
4. Quantize and pack with `fixed_preprocess_tensor(...)`.
5. Use block size = `PARALLELISM_DIM_0 * PARALLELISM_DIM_1`.
6. Pad final block to full block size when needed.
7. Append each block to the stream driver queue.

Expected runtime logs location:
- Cocotb simulator output stream in terminal while running tests.
- Same messages in simulator log output under the generated project run directory.

## Done criteria
- DRAM `top.sv` has parameter stream ports.
- TB runtime logs show expected ports, bound drivers, and queued DRAM blocks.
- BRAM tests still pass unchanged.
