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
    ../cast/rtl/floor_round.sv
    ../cast/rtl/signed_clamp.sv
    ../cast/rtl/fixed_signed_cast.sv
    ../fixed_arithmetic/rtl/fixed_accumulator.sv
    ../fixed_arithmetic/rtl/fixed_adder_tree.sv
    ../fixed_arithmetic/rtl/fixed_adder_tree_layer.sv
    ../fixed_arithmetic/rtl/fixed_lut_index.sv
    ../fixed_arithmetic/rtl/fixed_lut.sv
    ../fixed_arithmetic/rtl/fixed_nr_stage.sv
    ../fixed_arithmetic/rtl/fixed_range_augmentation.sv
    ../fixed_arithmetic/rtl/fixed_range_reduction.sv
    ../fixed_arithmetic/rtl/fixed_isqrt.sv
    ../matmul/rtl/matrix_fifo.sv
    ../matmul/rtl/matrix_flatten.sv
    ../matmul/rtl/matrix_unflatten.sv
    ../norm/rtl/group_norm_2d.sv
}

import_files -force

update_compile_order -fileset sources_1

start_gui
