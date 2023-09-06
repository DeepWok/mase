from .memory import RunnerAvgBitwidth

HW_RUNNERS = {
    "average_bitwidth": RunnerAvgBitwidth,
}


def get_hw_runner(name: str, model_info, task: str, dataset_info, accelerator):
    if name not in HW_RUNNERS:
        raise ValueError(f"Hardware runner {name} is not supported.")
    return HW_RUNNERS[name](model_info, task, dataset_info, accelerator)
