#! /usr/bin/env python3

from argparse import ArgumentParser
import sys, os, toml
from typing import Any

default_base = {
    "model": "toy",
    "dataset": "toy-tiny",
    "passes": {
        # "profile_statistics": {
        #     "by": "type",
        #     "target_weight_nodes": ["linear"],
        #     "target_activation_nodes": ["relu", "linear"],
        # },
        "quantize": {
            "by": "type",
            "report": True,
            "default": {
                "config": {
                    "name": "ternary",
                    "data_in_scaling_factor": True,
                    "data_in_width": 2,
                    "weight_scaling_factor": True,
                    "weight_width": 2,
                    "bias_scaling_factor": True,
                    "bias_width": 2,
                    "data_in_mean": "NA",
                    "data_in_median": "NA",
                    "data_in_max": "NA",
                    "weight_mean": "NA",
                    "weight_median": "NA",
                    "weight_max": "NA",
                    "bias_mean": "NA",
                    "bias_median": "NA",
                    "bias_max": "NA",
                }
            },
        },
    },
}


def set_stat(
    entry_name: str,
    mean: float | None | str = None,
    median: float | None | str = None,
    max: float | None | str = None,
) -> dict[str, Any]:
    """Return a dictionary containing the format of the stats required to use
    ternary quantiser. If statistics are not specified, "NA" will be set as the value,
    this interally is being interpreted as None when the .toml is loaded"""
    if mean == None:
        mean = "NA"
    if median == None:
        median = "NA"
    if max == None:
        max = "NA"
    return {
        "{}_mean".format(entry_name): mean,
        "{}_median".format(entry_name): median,
        "{}_max".format(entry_name): max,
    }


class Translator:
    def __init__(self, args):
        self.args = args
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def emit_config(self) -> int:
        """Read stats data and generate appropriate Mase config"""
        if self.args.base:
            config = toml.load(self.args.base)
        else:
            config = default_base
        source = toml.load(self.args.source)
        for block_name, block in source.items():
            if "seq_blocks_" in block_name:
                config["passes"]["quantize"][block_name] = (
                    {
                        "config": {
                            "name": "ternary",
                            "data_in_scaling_factor": True,
                            "data_in_width": 2,
                            "weight_scaling_factor": True,
                            "weight_width": 2,
                            "bias_scaling_factor": True,
                            "bias_width": 2,
                        }
                    }
                    if source[block_name]["common"]["mase_op"] in ["linear"]
                    else {  # compatability list
                        "config": {
                            "name": "ternary",
                            "data_in_scaling_factor": True,
                            "data_in_width": 2,
                        }
                    }
                )
                for data_type, stat in block["software"]["args"].items():
                    # should be data_in, weight or bias
                    if data_type == "data_in_0":
                        data_type = "data_in"
                    stat = stat["stat"]
                    config["passes"]["quantize"][block_name]["config"].update(
                        set_stat(
                            data_type,
                            median=stat["range_quantile"]["max"],
                            max=stat["range_min_max"]["max"],
                        )
                    )
        dest = open(self.args.destination, "w")
        toml.dump(config, dest)
        dest.close()
        return 0


# ---------- main function --------------
def main():
    USAGE = """Usage: 
stat-to-conf.py <stat.toml> <dest.toml> [--base base.toml]"""

    parser = ArgumentParser(usage=USAGE)
    parser.add_argument(
        "source",
        # action="store_true",
        # dest="run_all",
        # default=False,
        help="Path to the file containing the releblockant statistics",
    )
    parser.add_argument(
        "destination",
        # default="",
        # nargs="+",
        # dest="test_cases",
        help="Written config file, ready to be used for quantisation",
    )
    parser.add_argument(
        "-b",
        "--base",
        help="Specify a base config to create the new one. Stats will be appended",
    )

    args = parser.parse_args()
    translate = Translator(args)
    run = translate.emit_config()
    if run:
        sys.exit(run)
    sys.exit(-1)

    args = parser.parse_args()


if __name__ == "__main__":
    main()
