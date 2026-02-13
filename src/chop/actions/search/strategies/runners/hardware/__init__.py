from .memory import RunnerAvgBitwidth
from .dummy import RunnerHWDummy
from .resource import ResourceUsageRunner
from .latency import LatencyRunner

HW_RUNNERS = {
    "average_bitwidth": RunnerAvgBitwidth,
    "hw_dummy": RunnerHWDummy,
    "resource_usage": ResourceUsageRunner,
    "latency": LatencyRunner,
}


def get_hw_runner(name: str, model_info, task: str, dataset_info, accelerator, config):
    if name not in HW_RUNNERS:
        raise ValueError(f"Hardware runner {name} is not supported.")
    return HW_RUNNERS[name](model_info, task, dataset_info, accelerator, config)
