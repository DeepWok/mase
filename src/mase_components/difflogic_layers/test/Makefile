# Makefile for running the cocotb testbench on fixed_difflogic_logic

TOPLEVEL_LANG = verilog # keep verilog even though top level is sv

# Specify top_level and all subsidary modules used in tb
VERILOG_SOURCES = $(shell pwd)/../rtl/fixed_difflogic_logic_neuron.sv $(shell pwd)/../rtl/fixed_difflogic_logic.sv
TOPLEVEL = fixed_difflogic_logic
MODULE = fixed_difflogic_logic_tb
SIM ?= verilator

include $(shell cocotb-config --makefiles)/Makefile.sim
