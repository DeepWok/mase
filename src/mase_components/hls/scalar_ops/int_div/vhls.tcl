open_project -reset prj 
set_top div 
add_files div.cpp 
open_solution -reset "solution1"
set_part {xcu250-figd2104-2L-e}
create_clock -period 10 -name default
config_bind -effort high 
csynth_design
