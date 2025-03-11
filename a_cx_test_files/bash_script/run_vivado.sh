#/bin/bash
#python3 ./test/passes/graph/transforms/verilog/test_emit_verilog_$1.py
cd $1/hardware/top_build_project
vivado -mode batch -log project_build.log -source build.tcl
