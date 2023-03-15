import torch

config = dict(
    print_profile=True,
    detailed=True,
    module_depth=-1,
    top_modules=1,
    warm_up=10,
    as_string=True,
    output_file="estimate-sw_reports/opt-350_no_linear_layer.txt",
    ignore_modules=[torch.nn.Linear],
)
