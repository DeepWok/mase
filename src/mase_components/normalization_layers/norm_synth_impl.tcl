# Run this script with: vivado -mode tcl -source norm_synth_impl.tcl

# Script parameters
set fpga_part xcu250-figd2104-2L-e
set runs {
    batch_norm_2d
    group_norm_2d
    rms_norm_2d
}

# 2ns clk constraint for lower bitwidths
# set bitwidths {2 4 6 8}
# set constraints_file alveo-u250-2ns.xdc

# 4ns clk constraint for higher bitwidths
set bitwidths {10 12 14 16}
set constraints_file alveo-u250-4ns.xdc

# Shape Parameters
set total_dim0 64
set total_dim1 64
set compute_dim0 4
set compute_dim1 4
set channels 32


# Verilog Dependencies
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
    ../norm/rtl/rms_norm_2d.sv
    ../norm/rtl/channel_selection.sv
    ../norm/rtl/batch_norm_2d.sv
}

# Constraints file
read_xdc $constraints_file

# Vivado Synth/Impl loop
foreach top_module $runs {
    foreach width $bitwidths {

        set_property top $top_module [current_fileset]

        set frac_width [expr {$width / 2}]

        if {$top_module == "batch_norm_2d"} {
            synth_design -mode out_of_context -flatten_hierarchy rebuilt \
                         -top $top_module -part $fpga_part -debug_log \
                         -generic TOTAL_DIM0=$total_dim0 \
                         -generic TOTAL_DIM1=$total_dim1 \
                         -generic COMPUTE_DIM0=$compute_dim0 \
                         -generic COMPUTE_DIM1=$compute_dim1 \
                         -generic NUM_CHANNELS=$channels \
                         -generic IN_WIDTH=$width \
                         -generic IN_FRAC_WIDTH=$frac_width \
                         -generic OUT_WIDTH=$width \
                         -generic OUT_FRAC_WIDTH=$frac_width
        } elseif {$top_module == "rms_norm_2d"} {
            synth_design -mode out_of_context -flatten_hierarchy rebuilt \
                         -top $top_module -part $fpga_part -debug_log \
                         -generic TOTAL_DIM0=$total_dim0 \
                         -generic TOTAL_DIM1=$total_dim1 \
                         -generic COMPUTE_DIM0=$compute_dim0 \
                         -generic COMPUTE_DIM1=$compute_dim1 \
                         -generic CHANNELS=$channels \
                         -generic IN_WIDTH=$width \
                         -generic IN_FRAC_WIDTH=$frac_width \
                         -generic SCALE_WIDTH=$width \
                         -generic SCALE_FRAC_WIDTH=$frac_width \
                         -generic OUT_WIDTH=$width \
                         -generic OUT_FRAC_WIDTH=$frac_width
        } else {
            synth_design -mode out_of_context -flatten_hierarchy rebuilt \
                         -top $top_module -part $fpga_part -debug_log \
                         -generic TOTAL_DIM0=$total_dim0 \
                         -generic TOTAL_DIM1=$total_dim1 \
                         -generic COMPUTE_DIM0=$compute_dim0 \
                         -generic COMPUTE_DIM1=$compute_dim1 \
                         -generic GROUP_CHANNELS=$channels \
                         -generic IN_WIDTH=$width \
                         -generic IN_FRAC_WIDTH=$frac_width \
                         -generic OUT_WIDTH=$width \
                         -generic OUT_FRAC_WIDTH=$frac_width
        }

        write_checkpoint build/${top_module}/${width}bit/post_synth.dcp -force

        # Implementation
        opt_design
        place_design
        phys_opt_design
        route_design

        # Utilization report
        report_utilization -file build/${top_module}/${width}bit/utilization.rpt

        # Timing report
        report_timing_summary -delay_type min_max -check_timing_verbose \
                            -max_paths 100 -nworst 10 -input_pins -routable_nets \
                            -file build/${top_module}/${width}bit/timing.rpt

        write_checkpoint build/${top_module}/${width}bit/post_route.dcp -force
    }
}

exit
