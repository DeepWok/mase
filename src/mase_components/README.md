# Mase Hardware

This directory stores a list of hardware block templates in SystemVerilog:

* `common` includes a set of hardware blocks for basic functions 
* `linear` includes a set of hardware blocks for linear layers
* `activations` includes a set of hardware blocks for activation layers
* `testbench` includes the test benches for all the hardware blocks

## Detailed list of hardware blocks

* `common`
  * `accumulator`: a N-input handshaked accumulator
  * `adder_tree_layer`: a 2N-to-N adder array
  * `adder_tree`: a N-input handshaked adder tree
  * `int_mult`: a fixed-point multiplier
  * `vector_mult`: a fixed-point handshaked vector multiplier
  * `register_slice`: a single handshaked register
* `linear`
  * `dataflow_linear`: a handshaked linear layer
* `activations`
  * `int_relu`: a fixed-point relu

## To contribute

If a hardware block needs to be added, please follow the following procedures:

1. Add SystemVerilog code into a suitable directory
2. Format the code using `verible`: `verible-verilog-format --inplace xxx.sv`
3. Lint the code using `verilator`: `verilator --lint-only --Wall xxx.sv`
4. Implement a test bench using `cocotb` for the hardware block under `testbench`
5. Test the hardware block with the test bench using `test-hardware.py`: `test-hardware.py xxx`
6. Added this hardware block into the regression test list in `mase-tools/bin/test-hardware.py`
7. Update this `README.md`
