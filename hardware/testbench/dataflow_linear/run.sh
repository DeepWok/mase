#!/bin/sh

rm -rf obj_dir
rm -f .vcd

verilator -Wall --cc --trace ../../common/*.sv ../../linear/dataflow_linear.sv --top-module dataflow_linear --exe dataflow_linear_tb.cpp

make -j -C obj_dir/ -f Vdataflow_linear.mk Vdataflow_linear_tb

obj_dir/Vdataflow_linear_tb
