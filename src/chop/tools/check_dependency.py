import logging
from importlib.util import find_spec
from chop.passes.utils import PassFactory

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


def find_missing_dependencies(
    pass_name: str,
):
    dependencies = PassFactory._dependencies_dict.get(pass_name, None)

    if dependencies is None:
        return []

    availabilities = [find_spec(dep) is not None for dep in dependencies]
    unavailable_deps = [
        dep for dep, avail in zip(dependencies, availabilities) if not avail
    ]

    return unavailable_deps


def check_dependencies(
    pass_name: str,
    silent: bool = True,
):
    unavailable_deps = find_missing_dependencies(pass_name)

    if not silent:
        if len(unavailable_deps) > 0:
            logger.warning(
                f"Pass: {pass_name} is unavailable because the following dependencies are not installed: {', '.join(unavailable_deps)}."
            )
        else:
            logger.info(
                f"Extension: All dependencies for the {pass_name} pass are available."
            )

    return len(unavailable_deps) == 0
