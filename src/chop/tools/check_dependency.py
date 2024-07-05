import logging
from importlib.util import find_spec

logger = logging.getLogger(__name__)


def check_deps_tensorRT_pass(silent: bool = True):
    dependencies = ["pytorch_quantization", "tensorrt", "pynvml", "pycuda", "cuda"]

    availabilities = [find_spec(dep) is not None for dep in dependencies]
    unavailable_deps = [
        dep for dep, avail in zip(dependencies, availabilities) if not avail
    ]

    if not silent:
        if not all(availabilities):
            logger.warning(
                f"TensorRT pass is unavailable because the following dependencies are not installed: {', '.join(unavailable_deps)}."
            )
        else:
            logger.info("Extension: All dependencies for TensorRT pass are available.")
    return all(availabilities)
