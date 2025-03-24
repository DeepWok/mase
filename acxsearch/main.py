import sys
import time
from pathlib import Path
import torch
import argparse
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1].as_posix()
sys.path.append(ROOT)
sys.path.append(ROOT + "/machop")

from search_func import _in_layer_quant_search

from machop.chop.passes.graph.analysis import add_software_metadata_analysis_pass
from machop.chop.passes.graph.transforms.quantize import (
    quantize_transform_pass,
    softmax_transform_pass,
)
from a4cirrus.utils import acc_cal, loss_cal, initialize_graph, parse_config_choice
import yaml
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--load_name", default=None)
    parser.add_argument("--load_type", default="mz")
    parser.add_argument("--config")
    args = parser.parse_args()

    return args

def main(args):
    mase_path = Path(ROOT)
    ini_args = {
        "model_name": args.model_name,
        "dataset_name": args.datasets,
        "batch_size": args.batch_size,
        "load_name": args.load_name,
        "load_type": args.load_type,
    }

    mg, info = initialize_graph(**ini_args)
    mg, _ = add_software_metadata_analysis_pass(mg, None)
    breakpoint()
    args = vars(args)
    with open(args["config"], "r") as f:
        config = yaml.safe_load(f)
    
    parse_config_choice(config['config_choice'])
    for entry, value in args.items():
        if value is not None:
            config[entry] = value
    match config["task"]:
        case "quantile_int":
            quant_cfg = _in_layer_quant_search(mg, info, config)
            mg, _ = quantize_transform_pass(mg, quant_cfg)
        case "quantile_scale_int":
            quant_cfg = _in_layer_quant_search(mg, info, config)
            mg, _ = quantize_transform_pass(mg, quant_cfg)
        case "bfp":
            config["config_choice"] = {
                "act": [(8, 8, None, 32), (6, 8, None, 32), (4, 8, None, 32)],
                "w": [(8, 8, None, 32), (6, 8, None, 32), (4, 8, None, 32)],
                "b": [(8, 8, None, 32), (6, 8, None, 32), (4, 8, None, 32)],
            }
            config["config_list"] = [
                "width",
                "exponent_width",
                "exponent_bias",
                "block_size",
            ]
            quant_cfg = _in_layer_quant_search(mg, info, config)
            mg, _ = quantize_transform_pass(mg, quant_cfg)
        case "softmax":
            quant_cfg = _in_layer_quant_search(mg, info, config)
            mg, _ = softmax_transform_pass(mg, quant_cfg)
        case _:
            raise NotImplementedError(f"{args.task} is not supported right now")

    save_dir = "_".join(
        [
            datetime.now().strftime("%m_%d_%H"),
            config["model_name"],
            config["datasets"],
        ]
    )
    save_folder = Path(__file__).parent[0] /"output"/ save_dir
    save_folder.mkdir(parents=True, exist_ok=True)
    save_info = {}
    save_info["quant"] = quant_cfg
    with open(save_folder / "InLayerQuantSearch.json", "w") as f:
        json.dump(save_info, f, indent=4)

    test_dataloader = info["data_module"].test_dataloader()
    acc_after = acc_cal(mg.model, test_dataloader)
    save_info["acc_before"] = 0.85104
    save_info["acc_after"] = acc_after
    with open(save_folder / "InLayerQuantSearch.json", "w") as f:
        json.dump(save_info, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
