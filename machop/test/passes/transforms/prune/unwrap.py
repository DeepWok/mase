# NOTE: Before running this file, make sure that you have a model pruned via prune.py.
# The unwrap routine takes care of removing the activation handler along with all the
# pre-forward hooks that enforce or observe activation sparsity.

import logging
import os
import sys
from pathlib import Path

# Housekeeping -------------------------------------------------------------------------
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(Path(__file__).resolve().parents[4].as_posix())

from chop.passes import (
    add_mase_ops_analysis_pass,
    init_metadata_analysis_pass,
    prune_unwrap_transform_pass,
)
from chop.passes.graph.mase_graph import MaseGraph
from chop.tools.logger import getLogger

logger = getLogger("chop")
logger.setLevel(logging.DEBUG)


def main():
    # We don't want to run this script if there's no path provided
    if len(sys.argv) < 2:
        return

    # This path should point to a mase checkpoint!
    model_path = Path(sys.argv[1]).resolve().absolute()
    graph = MaseGraph(model=None, load_name=model_path.as_posix())
    # NOTE: Both functions have pass arguments that are not used in this example
    graph = init_metadata_analysis_pass(graph, None)
    graph = add_mase_ops_analysis_pass(graph, None)

    graph = prune_unwrap_transform_pass(graph, None, None)


if __name__ == "__main__":
    main()
