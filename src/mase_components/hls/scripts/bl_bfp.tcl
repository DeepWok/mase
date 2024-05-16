open_project -reset bl_bfp 
set_top top 
add_files bl_bfp.cpp 
open_solution -reset "solution1"
set_part {xcu250-figd2104-2L-e}
create_clock -period 10 -name default
config_bind -effort low 
config_compile -pipeline_loops 1
csynth_design
export_design -flow syn -format ip_catalog
