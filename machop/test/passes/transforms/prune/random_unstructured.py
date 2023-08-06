#!/usr/bin/env python3
# NOTE: This is not really a test, but a script to just informally validate
# functionality via trail and error. Feel free to modify this file as needed.

import logging
import os
import sys
from pathlib import Path

import toml

# Housekeeping -------------------------------------------------------------------------
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(Path(__file__).resolve().parents[4].as_posix())

from chop.passes import (
    add_mase_ops_analysis_pass,
    init_metadata_analysis_pass,
    prune_transform_pass,
)
from chop.passes.graph.mase_graph import MaseGraph
from chop.models.vision import get_mobilenetv3_small
from chop.tools.logger import getLogger


logger = getLogger("chop")
logger.setLevel(logging.DEBUG)


def main():
    model = get_mobilenetv3_small({"num_classes": 1000}, pretrained=True)
    graph = MaseGraph(model=model)

    # NOTE: Both functions have pass arguments that are not used in this example
    graph = init_metadata_analysis_pass(graph, None)
    graph = add_mase_ops_analysis_pass(graph, None)
    logger.debug(graph.fx_graph)

    root = Path(__file__).resolve().parents[4]
    config_path = root / "configs/tests/prune/random_unstructured.toml"
    with open(config_path) as f:
        config = toml.load(f)
        config = config["passes"]["prune"]

        graph = prune_transform_pass(graph, config)


if __name__ == "__main__":
    main()
