#/bin/bash
CONFIG_PATH=$1.yaml python3 test/passes/graph/transforms/verilog/test_emit_verilog_mxint_vit_folded_top.py 
CONFIG_PATH=$1.yaml python3 test/passes/graph/transforms/verilog/test_emit_verilog_mxint_real_top.py 
#cd /scratch/cx922/mase/mxint_$1/hardware/top_build_project
#vivado -mode batch -log project_build.log -source build.tcl
