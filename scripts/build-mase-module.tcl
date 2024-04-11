#!/usr/bin/env tclsh

# Set board repository
set_param board.repoPaths {/home/pg519/shared/board-files}

set group [lindex $argv 0]
set module [lindex $argv 1]

puts "group $group, module $module"

set project_name "vivado-build-$group-$module"

# Define paths
set script_path [file normalize [info script]]
set scripts_directory [file dirname $script_path]
set mase_components_directory [file join $scripts_directory "../machop/mase_components"]
# set group_directory [file join $mase_components_directory "/$group"]
puts "script_path: $script_path"
puts "scripts_directory: $scripts_directory"
puts "mase_components_directory: $mase_components_directory"
# puts "group_directory: $group_directory"

# create_project $project_name -part xcu280-fsvh2892-2L-e -force

# Create the build directory, ignoring any exception
catch {exec mkdir $project_name}
cd $project_name

# Check if the project exists
if {[file exists $project_name.xpr]} {
    puts "Build project already exists, opening it."
    open_project $project_name.xpr
} else {
    # Project doesn't exist, create a new project
    puts "Creating build project."
    create_project $project_name -part xcu280-fsvh2892-2L-e -force
    set_property board_part xilinx.com:au280:part0:1.1 [current_project]
}

add_files $mase_components_directory

set_property top $module [current_fileset]

launch_runs synth_1 -jobs 32
wait_on_runs -timeout 60 synth_1

launch_runs impl_1 -jobs 32
wait_on_runs -timeout 60 impl_1