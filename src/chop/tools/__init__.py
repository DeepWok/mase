# import os
# import pathlib

# DATA_DIR = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "data")
# os.environ["GG_DATA_DIR"] = DATA_DIR

from .checkpoint_load import load_model
from .config_load import load_config, post_parse_load_config
from .logger import root_logger, set_logging_verbosity, get_logger

from .get_input import get_cf_args, get_dummy_input

from .utils import (
    set_excepthook,
    deepsetattr,
    deepgetattr,
    get_checkpoint_file,
    copy_weights,
    to_numpy,
    to_numpy_if_tensor,
    to_tensor,
    to_tensor_if_numpy,
    is_tensor,
)
