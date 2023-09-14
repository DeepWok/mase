import os
import sys
from pathlib import Path
import logging

# Housekeeping -------------------------------------------------------------------------
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
root = Path(__file__).resolve().parents[5]
path = root / "machop/third-party/fpgaconvnet-optimiser"
assert path.exists(), "fpgaconvnet-optimiser not found!"
assert (path / "fpgaconvnet-model").exists(), "fpgaconvnet-model not found!"
sys.path.append((root / path).as_posix())
sys.path.append((root / path / "fpgaconvnet-model").as_posix())

logger = logging.getLogger(__name__)

try:
    import fpgaconvnet.optimiser.cli as cli
except ImportError:
    logger.warning(
        "Third-party package fpgaconvnet-optimiser is not installed properly. "
        "The pass `fpgaconvnet_optimiser_analysis_pass` will not work. "
        "Run `make sync-fpgaconvnet && ./scripts/init-conda-fpgaconvnet.sh` under mase-tools/ to it setup."
    )


def fpgaconvnet_optimiser_analysis_pass(graph, pass_args: dict):
    sys.argv = [
        __file__,
        "--name",
        pass_args["test_name"],
        "--model_path",
        pass_args["model_path"],
        "--platform_path",
        (path / "examples/platforms/u250.toml").as_posix(),
        "--output_path",
        (pass_args["save_dir"] / "fpgaconvnet_optimiser").as_posix(),
        "-b",
        "256",
        "--objective",
        "throughput",
        "--optimiser",
        "greedy_partition",
        "--optimiser_config_path",
        pass_args["optimiser_config_path"],
    ]
    cli.main()
    return graph
