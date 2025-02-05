# #!/usr/bin/env python3
# # This example converts a simple MLP model to Verilog
# import logging
# import os
# import sys

# import torch
# import torch.nn as nn

# from pathlib import Path

# sys.path.append(Path(__file__).resolve().parents[5].as_posix())


# import chop.passes as passes
# from chop.pipelines import AutoPipeline

# import torch
# from torch import nn
# from transformers import RobertaForSequenceClassification, AutoTokenizer

# pretrained = "XianYiyk/roberta-relu-pretrained-sst2"
# bert = RobertaForSequenceClassification.from_pretrained(pretrained, num_labels=2)
# tokenizer = AutoTokenizer.from_pretrained(pretrained, do_lower_case=True)
# for param in bert.parameters():
#     param.requires_grad = True  # QAT training


# class SpikeZIP_TF_Pipeline(AutoPipeline):
#     def __init__(self):
#         pass_list = [
#             passes.quantize_module_transform_pass,
#             passes.ann2snn_module_transform_pass,
#             passes.ann2snn_module_transform_pass,
#         ]
#         super().__init__([pass_list])


# pipe = SpikeZIP_TF_Pipeline()


# quan_pass_args = {
#     "by": "regex_name",
#     "roberta\.encoder\.layer\.\d+\.attention\.self": {
#         "config": {
#             "name": "lsqinteger",
#             "level": 32,
#         }
#     },
#     "roberta\.encoder\.layer\.\d+\.attention\.output": {
#         "config": {
#             "name": "lsqinteger",
#             "level": 32,
#         }
#     },
#     "roberta\.encoder\.layer\.\d+\.output": {
#         "config": {
#             "name": "lsqinteger",
#             "level": 32,
#         }
#     },
#     "roberta\.encoder\.layer\.\d+\.intermediate": {
#         "config": {
#             "name": "lsqinteger",
#             "level": 32,
#         }
#     },
#     "classifier": {
#         "config": {
#             "name": "lsqinteger",
#             "level": 32,
#         }
#     },
# }
# convert_pass_args_1 = {
#     "by": "regex_name",
#     "roberta\.encoder\.layer\.\d+\.attention\.self": {
#         "config": {
#             "name": "zip_tf",
#             "level": 32,
#             "neuron_type": "ST-BIF",
#         },
#     },
# }
# convert_pass_args_2 = {
#     "by": "type",
#     "embedding": {
#         "config": {
#             "name": "zip_tf",
#         },
#     },
#     "linear": {
#         "config": {
#             "name": "unfold_bias",
#             "level": 32,
#             "neuron_type": "ST-BIF",
#         },
#     },
#     "conv2d": {
#         "config": {
#             "name": "zip_tf",
#             "level": 32,
#             "neuron_type": "ST-BIF",
#         },
#     },
#     "layernorm": {
#         "config": {
#             "name": "zip_tf",
#         },
#     },
#     "relu": {
#         "manual_instantiate": True,
#         "config": {
#             "name": "identity",
#         },
#     },
#     "lsqinteger": {
#         "manual_instantiate": True,
#         "config": {
#             "name": "st_bif",
#             # Default values. These would be replaced by the values from the LSQInteger module, so it has no effect.
#             # "q_threshold": 1,
#             # "level": 32,
#             # "sym": True,
#         },
#     },
# }


# mg, _ = pipe(
#     bert,
#     pass_args={
#         "quantize_module_transform_pass": quan_pass_args,
#         "ann2snn_module_transform_pass": convert_pass_args_1,
#         "ann2snn_module_transform_pass": convert_pass_args_2,
#     },
# )
