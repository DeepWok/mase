# Run this script with: vivado -mode tcl -source norm_synth_impl.tcl

create_project norm norm_proj_dir -part xcu250-figd2104-2L-e -force

add_files {
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
    ../norm/rtl/rms_norm_2d.sv
    ../norm/rtl/norm.sv
    isqrt-16-lut.mem
    top.sv
}

import_files -force

update_compile_order -fileset sources_1

# add_files -fileset constrs_1 alveo-u250-norm.xdc

# set_property target_constrs_file /scratch/ddl20/mase/machop/mase_components/norm/alveo-u250-norm.xdc [current_fileset -constrset]
read_xdc alveo-u250-norm.xdc

launch_runs synth_1 -jobs 10
wait_on_runs synth_1

launch_runs impl_1 -jobs 10
wait_on_runs impl_1

start_gui
