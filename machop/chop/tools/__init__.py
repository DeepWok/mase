# import os
# import pathlib

# DATA_DIR = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), "data")
# os.environ["GG_DATA_DIR"] = DATA_DIR

from .config_load import post_parse_load_config, load_config
from .checkpoint_load import check_when_to_load_and_how_to_load, load_model

from .logger import getLogger