# Makefile for running the cocotb testbench on difflogic_layers

TOPLEVEL_LANG = verilog # keep verilog even though top level is sv

# Specify top_level and all subsidary modules used in tb
VERILOG_SOURCES = $(shell pwd)/../../rtl/*.sv

# TOPLEVEL = fixed_difflogic_flatten
# TOPLEVEL = fixed_difflogic_groupsum
TOPLEVEL = fixed_difflogic_logic

# MODULE = fixed_difflogic_flatten_tb
# MODULE = fixed_difflogic_groupsum_tb
MODULE = fixed_difflogic_logic_tb

SIM ?= verilator

include $(shell cocotb-config --makefiles)/Makefile.sim
