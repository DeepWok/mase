import logging

from colorlog import ColoredFormatter

formatter = ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
    secondary_log_colors={},
    style="%",
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

root_logger = logging.getLogger("chop")
root_logger.addHandler(handler)


def set_logging_verbosity(level: str = "info"):
    level = level.lower()
    match level:
        case "debug":
            root_logger.setLevel(logging.DEBUG)
        case "info":
            root_logger.setLevel(logging.INFO)
        case "warning":
            root_logger.setLevel(logging.WARNING)
        case "error":
            root_logger.setLevel(logging.ERROR)
        case "critical":
            root_logger.setLevel(logging.CRITICAL)
        case _:
            raise ValueError(
                f"Unknown logging level: {level}, should be one of: debug, info, warning, error, critical"
            )
    root_logger.info(f"Set logging level to {level}")


def get_logger(name: str):
    return root_logger.getChild(name)
