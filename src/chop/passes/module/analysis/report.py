import torch
import torch.nn as nn
from tabulate import tabulate


def get_submodule_summary(name: str, module: nn.Module, level: int = 0):
    submodule_summary = []
    total_params = 0

    for child_name, child_module in module.named_children():
        child_full_name = f"{name}.{child_name}" if name else child_name
        child_params = sum(
            p.numel() for p in child_module.parameters() if p.requires_grad
        )
        submodule_summary.append((level, child_full_name, child_params))
        total_params += child_params

        # Recursively get summaries for child modules
        child_summary, child_total = get_submodule_summary(
            child_full_name, child_module, level + 1
        )
        submodule_summary.extend(child_summary)
        total_params += child_total

    return submodule_summary, total_params


def report_trainable_parameters_analysis_pass(
    module: torch.nn.Module,
    pass_args: dict = {},
):
    submodule_summary, total_params = get_submodule_summary("", module)
    table = [(name, params) for _, name, params in submodule_summary]

    print(
        tabulate(table, headers=["Submodule", "Trainable Parameters"], tablefmt="grid")
    )
    print(f"\nTotal Trainable Parameters: {total_params}")

    return module, {}
