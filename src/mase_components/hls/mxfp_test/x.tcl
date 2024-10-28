open_project -reset x0_mxfp_linear2d_16_1_8_3_2_16_8_3 
set_top mxfp_linear2d_0
add_files { test.cpp }
open_solution -reset "solution1"
set_part {xcu250-figd2104-2L-e}
create_clock -period 4 -name default
config_bind -effort high
config_compile -pipeline_loops 1
csynth_design
