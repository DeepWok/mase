from importlib.util import find_spec
from chop.tools import get_logger

logger = get_logger(__name__)
logger.setLevel("INFO")


class PassFactory:
    _pass_fn_dict: dict = {}
    _dependencies_dict: dict = {}
    _requires_nightly_torch_dict: dict = {}

    def __init__(self):
        pass


def _nightly_torch_installed():
    try:
        import torch

        return "dev" in torch.__version__
    except ImportError:
        return False


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


def register_mase_pass(
    name: str,
    dependencies: list = [],
    requires_nightly_torch: bool = False,
):
    """This decorator registers a mase pass as PassFactory class attributes which can be used globally."""

    def decorator(fn):
        PassFactory._pass_fn_dict[name] = fn
        PassFactory._dependencies_dict[name] = dependencies
        PassFactory._requires_nightly_torch_dict[name] = requires_nightly_torch

        def wrapped_fn(*args, **kwargs):
            missing_deps = find_missing_dependencies(name)
            if missing_deps:
                logger.warning(
                    f"Missing dependencies for '{name}': {', '.join(missing_deps)}.\nThe function may not work as expected.",
                )
            if requires_nightly_torch and not _nightly_torch_installed():
                logger.warning(
                    f"Pass '{name}' requires a nightly version of PyTorch, but it is not installed.\nThe function may not work as expected."
                )
            return fn(*args, **kwargs)

        return wrapped_fn

    return decorator
