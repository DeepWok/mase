from .software import get_sw_runner


def get_hw_runner(*args, **kwargs):
    raise ValueError(
        "Hardware search runners have been removed. Configure only `sw_runner`."
    )
