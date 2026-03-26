# Use Verilator
SIM ?= verilator
TOPLEVEL_LANG ?= verilog

# Pass extra arguments to Verilator
# -Wno-fatal: Prevents simulation from stopping on minor linting warnings
# --trace: Enables waveform dumping (.vcd) for debugging
EXTRA_ARGS += -Wno-fatal --trace

# Grab ALL SystemVerilog files in the rtl directory
# Notice the extra ../ since we are now in test/mase_top_tb/
VERILOG_SOURCES += $(wildcard $(PWD)/../../rtl/*.sv)

# TOPLEVEL is the name of the top-level module in your Verilog file.
TOPLEVEL = top

# MODULE is the name of your Python test file (without the .py extension).
# CHANGE THIS to match the actual name of your Python file!
MODULE = test

# Include the Cocotb makefiles
include $(shell cocotb-config --makefiles)/Makefile.sim