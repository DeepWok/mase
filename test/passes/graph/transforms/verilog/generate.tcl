source config.tcl

create_project -in_memory -part xcku5p-ffvb676-2-e
set_property board_part xilinx.com:kcu116:part0:1.5 [current_project]

add_files -fileset sources_1 "$top_dir/hardware/rtl/"

set_property top top [current_fileset]

puts "Trial: ${trial_number}"

eval "synth_design -mode out_of_context -top top -part xcku5p-ffvb676-2-e"

save_project_as -force my_project

launch_runs synth_1 -jobs 12
wait_on_run synth_1

open_run synth_1
report_utilization -file "$mase_dir/resources/util_${trial_number}.txt"
