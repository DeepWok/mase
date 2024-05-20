"""
Here the ignore_modules should include the subclasses of nn.Modules to be ignored.
Note that the FLOP profiler iterates all nn.Modules.
The profile of the parent module still accumulates child modules profiles if only the child module classes are put in the ignore list.
Thus to avoid profiling a specific nn.Module subclass, you also need to add all parent modules containing this subclass to the ignore list.
See the example config `roberta_no_linear.py`
"""

config = dict(
    print_profile=True,
    detailed=True,
    module_depth=-1,
    top_modules=3,
    warm_up=10,
    as_string=True,
    output_file="estimate-sw_reports/report_name.txt",
    ignore_modules=[],
)
