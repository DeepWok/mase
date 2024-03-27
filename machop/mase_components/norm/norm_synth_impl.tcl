# Run this script with: vivado -mode tcl -source norm_synth_impl.tcl

# create_project norm norm_proj_dir -part xcu250-figd2104-2L-e -force

read_verilog -sv {
    ../common/rtl/join2.sv
    ../common/rtl/split2.sv
    ../common/rtl/repeat_circular_buffer.sv
    ../common/rtl/skid_buffer.sv
    ../common/rtl/register_slice.sv
    ../common/rtl/simple_dual_port_ram.sv
    ../common/rtl/fifo.sv
    ../common/rtl/lut.sv
    ../cast/rtl/floor_round.sv
    ../cast/rtl/signed_clamp.sv
    ../cast/rtl/fixed_signed_cast.sv
    ../fixed_arithmetic/rtl/fixed_accumulator.sv
    ../fixed_arithmetic/rtl/fixed_adder_tree.sv
    ../fixed_arithmetic/rtl/fixed_adder_tree_layer.sv
    ../fixed_arithmetic/rtl/fixed_lut_index.sv
    ../fixed_arithmetic/rtl/fixed_nr_stage.sv
    ../fixed_arithmetic/rtl/fixed_range_augmentation.sv
    ../fixed_arithmetic/rtl/fixed_range_reduction.sv
    ../fixed_arithmetic/rtl/fixed_isqrt.sv
    ../matmul/rtl/matrix_fifo.sv
    ../matmul/rtl/matrix_flatten.sv
    ../matmul/rtl/matrix_unflatten.sv
    ../norm/rtl/group_norm_2d.sv
}
# ../norm/rtl/rms_norm_2d.sv
# ../norm/rtl/norm.sv
# top.sv

read_xdc alveo-u250-norm.xdc

update_compile_order

# Synthesis
set_msg_config -id "Synth 8-3332" -limit 10000

synth_design -mode out_of_context -flatten_hierarchy rebuilt -top group_norm_2d -part xcu250-figd2104-2L-e -debug_log

write_checkpoint post_synth.dcp -force

# Implementation
opt_design
place_design
phys_opt_design
route_design

# Utilization report
report_utilization -file utilization.rpt

# Timing report
report_timing_summary -delay_type min_max -check_timing_verbose -max_paths 100 -nworst 10 -input_pins -routable_nets -file timing.rpt

write_checkpoint post_route.dcp -force

exit
