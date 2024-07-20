# Run this script with: vivado -mode tcl -source norm_synth_impl.tcl

# Script parameters
set fpga_part xcu250-figd2104-2L-e
set top_module fixed_grouped_query_attention

# 4ns clk constraint for higher bitwidths
set constraints_file alveo-u250.xdc

# Basic
# set sequence_len 16
# set embedding_dim 128
# set num_heads 8
# set num_kv_heads 4

# Realistic
# set sequence_len 128
# set embedding_dim 128
# set num_heads 8

# Heads Mem
# set sequence_len 256
# set embedding_dim 256
# set num_heads 16

# Llama 160m
set sequence_len 16
set embedding_dim 768
set num_heads 12
# set num_kv_heads 8

# Module Parameters
set num_kv_heads_list {1 2 3 4 6 12}
set embedding_par_list {16}
set sequence_par_list {1}

# Bitwidths
set bitwidths {8}

# Verilog Dependencies
read_verilog -sv {
    ../../arbiters/rtl/find_first_arbiter.sv
    ../../common/rtl/mux.sv
    ../../common/rtl/join2.sv
    ../../common/rtl/split2.sv
    ../../common/rtl/split_n.sv
    ../../common/rtl/join_n.sv
    ../../common/rtl/repeat_circular_buffer.sv
    ../../common/rtl/single_element_repeat.sv
    ../../common/rtl/skid_buffer.sv
    ../../common/rtl/unpacked_skid_buffer.sv
    ../../common/rtl/register_slice.sv
    ../../common/rtl/simple_dual_port_ram.sv
    ../../common/rtl/fifo.sv
    ../../common/rtl/lut.sv
    ../../common/rtl/comparator_tree.sv
    ../../common/rtl/comparator_accumulator.sv
    ../../common/rtl/register_slice.sv
    ../../common/rtl/unpacked_register_slice.sv
    ../../cast/rtl/fixed_round.sv
    ../../cast/rtl/fixed_rounding.sv
    ../../cast/rtl/floor_round.sv
    ../../cast/rtl/signed_clamp.sv
    ../../cast/rtl/fixed_signed_cast.sv
    ../../fixed_arithmetic/rtl/fixed_mult.sv
    ../../fixed_arithmetic/rtl/fixed_vector_mult.sv
    ../../fixed_arithmetic/rtl/fixed_dot_product.sv
    ../../fixed_arithmetic/rtl/fixed_accumulator.sv
    ../../fixed_arithmetic/rtl/fixed_adder_tree_layer.sv
    ../../fixed_arithmetic/rtl/fixed_adder_tree.sv
    ../../fixed_arithmetic/rtl/fixed_range_reduction.sv
    ../../linear/rtl/fixed_linear.sv
    ../../matmul/rtl/matrix_flatten.sv
    ../../matmul/rtl/matrix_unflatten.sv
    ../../matmul/rtl/matrix_fifo.sv
    ../../matmul/rtl/matrix_accumulator.sv
    ../../matmul/rtl/simple_matmul.sv
    ../../matmul/rtl/matmul.sv
    ../../matmul/rtl/transpose.sv
    ../../matmul/rtl/matrix_stream_transpose.sv
    ../../activations/rtl/softermax_lpw_pow2.sv
    ../../activations/rtl/softermax_lpw_reciprocal.sv
    ../../activations/rtl/softermax_local_window.sv
    ../../activations/rtl/softermax_global_norm.sv
    ../../activations/rtl/fixed_softermax_1d.sv
    ../../activations/rtl/fixed_softermax.sv
    ../../attention/rtl/fixed_gqa_projections.sv
    ../../attention/rtl/self_attention_head_single_scatter.sv
    ../../attention/rtl/gqa_head_scatter_control.sv
    ../../attention/rtl/self_attention_head_gather.sv
    ../../attention/rtl/fixed_self_attention_head.sv
    ../../attention/rtl/fixed_grouped_query_attention.sv
}

# Constraints file
read_xdc $constraints_file

# Vivado Synth/Impl

foreach sequence_parallelism $sequence_par_list {
    foreach embedding_parallelism $embedding_par_list {
        foreach num_kv_heads $num_kv_heads_list {
            foreach width $bitwidths {

                set frac_width [expr {$width / 2}]

                set_property top $top_module [current_fileset]

                # Write the config of the file
                set now [clock seconds]
                set timestr [clock format $now -format "%y-%m-%d-%H-%M-%S"]
                set working_dir build/${top_module}/${timestr}
                file mkdir $working_dir
                set config_fp [open "${working_dir}/config.json" w+]
                set config_str "{\n\
                    \"num_heads\": $num_heads,\n\
                    \"num_kv_heads\": $num_kv_heads,\n\
                    \"embedding_dim\": $embedding_dim,\n\
                    \"sequence_len\": $sequence_len,\n\
                    \"embedding_parallelism\": $embedding_parallelism,\n\
                    \"sequence_parallelism\": $sequence_parallelism,\n\
                    \"width\": $width,\n\
                    \"frac_width\": $frac_width\n\
                }"
                puts $config_fp $config_str
                close $config_fp

                synth_design -mode out_of_context \
                            -flatten_hierarchy rebuilt \
                            -top $top_module \
                            -part $fpga_part \
                            -debug_log \
                            -generic NUM_HEADS=$num_heads \
                            -generic NUM_GROUPS=$num_kv_heads \
                            -generic DATA_IN_0_TENSOR_SIZE_DIM_0=$embedding_dim \
                            -generic DATA_IN_0_TENSOR_SIZE_DIM_1=$sequence_len \
                            -generic DATA_IN_0_PARALLELISM_DIM_0=$embedding_parallelism \
                            -generic DATA_IN_0_PARALLELISM_DIM_1=$sequence_parallelism \
                            -generic DATA_IN_0_PRECISION_0=$width \
                            -generic DATA_IN_0_PRECISION_1=$frac_width \
                            -generic WEIGHT_TENSOR_SIZE_DIM_0=$embedding_dim \
                            -generic WEIGHT_TENSOR_SIZE_DIM_1=$embedding_dim \
                            -generic WEIGHT_PARALLELISM_DIM_0=$embedding_parallelism \
                            -generic WEIGHT_PARALLELISM_DIM_1=$embedding_parallelism \
                            -generic WEIGHT_PRECISION_0=$width \
                            -generic WEIGHT_PRECISION_1=$frac_width \
                            -generic WEIGHTS_PRE_TRANSPOSED=1 \
                            -generic HAS_BIAS=0

                # Post synthesis checkpoint
                write_checkpoint ${working_dir}/post_synth.dcp -force

                # Implementation
                opt_design
                place_design
                phys_opt_design
                route_design

                # Utilization report
                report_utilization -file ${working_dir}/utilization.rpt

                # Timing report
                report_timing_summary -delay_type min_max -check_timing_verbose \
                                    -max_paths 100 -nworst 10 -input_pins -routable_nets \
                                    -file ${working_dir}/timing.rpt

                write_checkpoint ${working_dir}/post_route.dcp -force

            }
        }
    }
}
exit
