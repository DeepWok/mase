import os

from .deps import MASE_HW_DEPS


def get_modules():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mods = [
        d
        for d in os.listdir(current_dir)
        if os.path.isdir(os.path.join(current_dir, d))
    ]
    if "__pycache__" in mods:
        mods.remove("__pycache__")
    return mods


def get_group_files(group):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    group_dir = os.path.join(current_dir, group, "rtl")
    files = [
        f"{group}/rtl/{f}"
        for f in os.listdir(group_dir)
        if os.path.isfile(os.path.join(group_dir, f)) and "__init__" not in f
    ]
    return files


def get_module_dependencies(module):
    group, mod = module.split("/")
    group_deps = MASE_HW_DEPS.get(module, [])
    file_deps = []
    for group_dep in group_deps:
        file_deps += get_group_files(group_dep)
    return file_deps
