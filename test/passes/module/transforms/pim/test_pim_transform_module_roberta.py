#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog

import torch
import torch.nn as nn

from pathlib import Path

from chop.passes.module.transforms import pim_matmul_transform_pass

import torch
from torch import nn
from transformers import RobertaForSequenceClassification, AutoTokenizer
import yaml

HOME_PATH = Path(__file__).resolve().parents[5].as_posix()

import copy

def test_pim_transform_module_roberta():
    pretrained = "JeremiahZ/roberta-base-mnli"
    model = RobertaForSequenceClassification.from_pretrained(
        pretrained, num_labels=2, ignore_mismatched_sizes=True
    )

    configs = ["pcm.yaml", "reram.yaml", "sram.yaml"]

    def get_model_summary(m):
        summary = {}
        for name, module in m.named_modules():
            class_name = module.__class__.__name__
            summary[class_name] = summary.get(class_name, 0) + 1
        return summary

    original_summary = get_model_summary(model)
    print("Original Model Summary (Layer Counts):")
    for cls, count in original_summary.items():
        if count > 1: # Only print interesting ones
            print(f"  {cls}: {count}")

    for config_name in configs:
        print(f"\n--- Testing with config: {config_name} ---")
        config_path = Path(HOME_PATH) / "configs" / "pim" / config_name
        with open(config_path, "r") as f:
            q_config = yaml.safe_load(f)

        # Use a deepcopy to avoid modifying the original model in each iteration
        model_to_transform = copy.deepcopy(model)
        qmodel, _ = pim_matmul_transform_pass(model_to_transform, q_config)
        
        q_summary = get_model_summary(qmodel)
        print(f"Transformed Model ({config_name}) Summary:")
        for cls, count in q_summary.items():
            if "PIM" in cls:
                print(f"  {cls}: {count}")

        # Print a few specific differences from the config
        print(f"  Key Config Parameters:")
        if "linear" in q_config:
            print(f"    Linear tile_type: {q_config['linear']['config'].get('tile_type')}")
            print(f"    Linear core_size: {q_config['linear']['config'].get('core_size')}")
        
        print(f"Successfully transformed with {config_name}")

if __name__ == "__main__":
    test_pim_transform_module_roberta()
