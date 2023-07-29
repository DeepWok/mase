DSE_MODES = ["codegen", "synth", "report", "all"]


def get_tcl_buff(project=None, top=None, cpp=None):
    assert project is not None
    assert top is not None
    assert cpp is not None
    return f"""
open_project -reset {project} 
set_top {top}
add_files {{ {cpp} }}
open_solution -reset "solution1"
set_part {{xcu250-figd2104-2L-e}}
create_clock -period 4 -name default
config_bind -effort high
config_compile -pipeline_loops 1
csynth_design
"""
