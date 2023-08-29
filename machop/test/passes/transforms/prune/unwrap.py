# NOTE: Before running this file, make sure that you have a model pruned via prune.py.
# The unwrap routine takes care of removing the activation handler along with all the
# pre-forward hooks that enforce or observe activation sparsity.

import logging
import os
import sys
import toml
from pathlib import Path

# Housekeeping -------------------------------------------------------------------------
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(Path(__file__).resolve().parents[4].as_posix())

from chop.passes import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    prune_unwrap_transform_pass,
    add_software_metadata_analysis_pass,
)
from chop.dataset import MaseDataModule
from chop.passes.graph.mase_graph import MaseGraph
from chop.tools.logger import getLogger
from chop.tools.get_input import get_dummy_input

logger = getLogger("chop")
logger.setLevel(logging.DEBUG)


def main():
    BATCH_SIZE = 32
    # We don't want to run this script if there's no path provided
    if len(sys.argv) < 2:
        return

    root = Path(__file__).resolve().parents[5]
    config_path = root / f"machop/configs/tests/prune/unwrap.toml"
    with open(config_path) as f:
        config = toml.load(f)

        # NOTE: We're only concerned with pre-trained vision models
        datamodule = MaseDataModule(
            model_name=config["model"],
            name=config["dataset"],
            batch_size=BATCH_SIZE,
            num_workers=os.cpu_count(),
            tokenizer=None,
            max_token_len=None,
        )
        datamodule.prepare_data()
        datamodule.setup()
        # NOTE: We only support vision classification models for now.
        dummy_input = get_dummy_input(datamodule, "cls", is_nlp_model=False)

        # This path should point to a mase checkpoint!
        model_path = Path(sys.argv[1]).resolve().absolute()
        graph = MaseGraph(model=None, load_name=model_path.as_posix())
        # NOTE: Both functions have pass arguments that are not used in this example
        graph = init_metadata_analysis_pass(graph, None)
        graph = add_common_metadata_analysis_pass(graph, dummy_input)
        graph = add_software_metadata_analysis_pass(graph, None)

        graph = prune_unwrap_transform_pass(graph, None, None)


if __name__ == "__main__":
    main()
