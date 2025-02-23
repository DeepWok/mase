
# set_param board.repoPaths {/home/cx922/shared/board-files}
create_project -force top_build_project mxint_vit_block/hardware/top_build_project -part xcu250-figd2104-2L-e
set_property board_part xilinx.com:au250:part0:1.3 [current_project]

add_files mxint_vit_block/hardware/rtl

set_property top top [current_fileset]
add_files /scratch/cx922/mase/src/mase_components/vivado/constraints.xdc
read_xdc /scratch/cx922/mase/src/mase_components/vivado/constraints.xdc

update_compile_order -fileset sources_1

launch_runs synth_1 -jobs 10
wait_on_run synth_1

launch_runs impl_1 -jobs 10
wait_on_run impl_1

ipx::package_project -root_dir mxint_vit_block/hardware/ip_repo -vendor user.org -library user -taxonomy /UserIP -import_files
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::check_integrity [ipx::current_core]
ipx::save_core [ipx::current_core]
set_property  ip_repo_paths  mxint_vit_block/hardware/ip_repo [current_project]
update_ip_catalog
