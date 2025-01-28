#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys

from chop.passes.module.transforms.snn.ann2snn import ann2snn_module_transform_pass
import torch
import torch.nn as nn

from pathlib import Path

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from accelerate import dispatch_model, infer_auto_device_map

sys.path.append(Path(__file__).resolve().parents[5].as_posix())


from chop.passes.module.transforms import quantize_module_transform_pass

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
import datasets as hf_datasets

import logging
import math

from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

# taken from:
# https://github.com/ChengZhang-98/lqer/blob/master/src/lqer/evaluate/evaluate_lm.py


def get_raw_data_module_wikitext2() -> hf_datasets.DatasetDict:
    dataset_dict = hf_datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    return dataset_dict


def preprocess_data_module_wikitext2(
    raw_dataset_dict,
    tokenizer,
    max_length,
    num_proc: int,
) -> hf_datasets.DatasetDict:
    if tokenizer.pad_token in ["<unk>", None]:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(["\n\n".join(examples["text"])])

    encodings = raw_dataset_dict.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_dataset_dict["train"].column_names,
        desc="Running tokenizer on dataset",
        num_proc=num_proc,
    )

    def group_texts(examples):
        # Concatenate all texts.
        # >>> sum([[1,2,3],[4,5,6]],[])
        # [1, 2, 3, 4, 5, 6]
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    preprocessed = encodings.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc="Grouping texts",
    )

    return preprocessed


def create_device_map(model, device_map) -> dict[str, int]:
    if device_map == "auto":
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=model._no_split_modules,
        )
    elif device_map == "auto-balanced":
        max_memory = {
            i: torch.cuda.mem_get_info(i)[0] // 2
            for i in range(torch.cuda.device_count())
        }
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=model._no_split_modules,
            max_memory=max_memory,
        )
        n_devices = torch.cuda.device_count()
        n_decoder_layers = model.config.num_hidden_layers
        n_layers_per_device = n_decoder_layers // n_devices
        balanced_device_map = {}
        current_device = 0
        current_decoder_idx = 0

        for layer_name in device_map:
            if ".layers." in layer_name:
                if (current_decoder_idx + 1) % n_layers_per_device == 0:
                    current_device += 1
                current_decoder_idx += 1
            balanced_device_map[layer_name] = min(current_device, n_devices - 1)
        device_map = balanced_device_map
    else:
        assert isinstance(device_map, dict)
    return device_map


def evaluate_perplexity(
    model,
    eval_dataloader: DataLoader,
    num_samples: int = None,
    progress_bar: bool = False,
    input_device: str = None,
    description: str = "Evaluating perplexity",
):
    if num_samples is not None:
        if not num_samples >= eval_dataloader.batch_size:
            txt = f"num_samples {num_samples} must be greater than batch_size {eval_dataloader.batch_size}"
            raise ValueError(txt)
        if not num_samples <= eval_dataloader.batch_size * len(eval_dataloader):
            txt = (
                f"num_samples {num_samples} must be less than or equal to "
                f"batch_size * len(eval_dataloader) = "
                f"{eval_dataloader.batch_size} * {len(eval_dataloader)} = {eval_dataloader.batch_size * len(eval_dataloader)}"
            )
            raise ValueError(txt)

    losses = []
    model.eval()

    # if input_device is None:
    #     input_device = model.device
    if num_samples:
        num_batches = num_samples // eval_dataloader.batch_size
    else:
        num_batches = len(eval_dataloader)

    progress_bar = tqdm(
        eval_dataloader,
        desc=description,
        total=num_batches,
        disable=not progress_bar,
    )

    batch_size = eval_dataloader.batch_size
    seq_len = next(iter(eval_dataloader))["input_ids"].shape[1]
    evaluated_samples = 0
    for i, batch in enumerate(eval_dataloader):
        if num_samples and i >= num_batches:
            break

        assert (
            batch["input_ids"].shape[1] == seq_len
        ), f"sequence length is not a constant current seq_len = {batch['input_ids'].shape[1]} != {seq_len}"
        with torch.no_grad():
            if input_device is None:
                input_device = next(iter(model.state_dict().values())).device
            batch = {
                k: v.to(input_device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = model(**batch)
        loss = outputs.loss.item() * batch_size * seq_len
        losses.append(loss)
        evaluated_samples += batch_size

        progress_bar.update(1)

    logger.debug(f"evaluated_samples = {evaluated_samples}")

    reduced_loss = sum(losses) / (seq_len * evaluated_samples)
    try:
        perplexity = math.exp(reduced_loss)
    except OverflowError:
        perplexity = float("inf")

    results = {
        "loss": reduced_loss,
        "perplexity": perplexity,
        "num_samples": evaluated_samples,
        "seq_len": seq_len,
        "batch_size": batch_size,
    }
    return results


# pretrained = "TinyLlama/TinyLlama_v1.1"
pretrained = "SparseLLM/ReluLLaMA-7B"
max_length = 512
num_proc = 1
batch_size = 8
num_samples = 100

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained)
tiny_model = AutoModelForCausalLM.from_pretrained(
    pretrained, torch_dtype=torch.float32, _attn_implementation="eager"
)

# Move the model to the appropriate device
device_map_ = create_device_map(tiny_model, device_map="auto")
tiny_model = dispatch_model(tiny_model, device_map=device_map_)
print("device:", tiny_model.device)

# Load the raw data module and preprocess it
data_module = get_raw_data_module_wikitext2()
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
dataset = preprocess_data_module_wikitext2(
    data_module, tokenizer, max_length, num_proc=num_proc
)
data_loader = torch.utils.data.DataLoader(
    dataset["validation"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator,
)

# results: {'loss': 2.2041055858135223, 'perplexity': 9.062142633092924, 'num_samples': 96, 'seq_len': 512, 'batch_size': 8}
results = evaluate_perplexity(
    tiny_model, data_loader, num_samples=num_samples, progress_bar=True
)
print(f"results: {results}")

for name, module in tiny_model.named_modules():
    print(name)


def test_ann2snn_module_transform_pass():
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
    quan_pass_args = {
        "by": "regex_name",
        "model.layers\.\d+\.self_attn": {
            "config": {
                "name": "lsqinteger",
                "level": 32,
            }
        },
    }

    mg, _ = quantize_module_transform_pass(tiny_model, quan_pass_args)

    return mg


mg = test_ann2snn_module_transform_pass()
device_map_ = create_device_map(mg, device_map="auto")
tiny_model = dispatch_model(mg, device_map=device_map_)


results = evaluate_perplexity(
    mg, data_loader, num_samples=num_samples, progress_bar=True
)
print(f"results: {results}")


# # Iterate through model layers and plot weights

# output_dir = "layer_weight_plots"
# os.makedirs(output_dir, exist_ok=True)
# exclude_layers = ["model.embed_tokens.weight"]

# for name, param in model.named_parameters():
#     if "weight" in name and all([exclude not in name for exclude in exclude_layers]):
#         # Move weights to GPU and calculate absolute values
#         weights = abs(param)

#         # Compute the outlier threshold on the GPU
#         outlier_threshold = weights.mean() + 3 * weights.std()

#         # Assign colors based on the outlier threshold
#         color_indices = torch.where(weights > outlier_threshold, 1, 0)

#         # Detach weights and move to CPU for plotting
#         weights = weights.detach().cpu().numpy().flatten()  # Flatten weights
#         color_indices = (
#             color_indices.detach().cpu().numpy().flatten()
#         )  # Flatten color indices
#         outlier_threshold = outlier_threshold.item()  # Convert to scalar

#         # Map color indices to actual colors
#         colors = ["red" if idx == 1 else "yellow" for idx in color_indices]

#         print(f"Layer: {name}")
#         # Create plots
#         plt.figure(figsize=(10, 6))
#         plt.scatter(range(len(weights)), weights, c=colors, s=1, alpha=0.7)
#         plt.title(f"Weight Values for {name}")
#         plt.xlabel("Weight Index")
#         plt.ylabel("Weight Value")
#         plt.ylim(0, max(weights) + 0.1)
#         plt.axhline(
#             outlier_threshold,
#             color="blue",
#             linestyle="--",
#             linewidth=1,
#             label="Outlier Threshold",
#         )
#         plt.legend(loc="upper right")

#         # Adjust layout and save the plot
#         plt.tight_layout()
#         plot_path = os.path.join(output_dir, f"{name.replace('.', '_')}_weights.png")
#         plt.savefig(plot_path)
#         plt.close()  # Close the plot to free memory


# print(f"Plots saved in directory: {output_dir}")


# for param in bert.parameters():
#     param.requires_grad = True  # QAT training


# def test_ann2snn_module_transform_pass():
#     quan_pass_args = {
#         "by": "regex_name",
#         "roberta\.encoder\.layer\.\d+\.attention\.self": {
#             "config": {
#                 "name": "lsqinteger",
#                 "level": 32,
#             }
#         },
#         "roberta\.encoder\.layer\.\d+\.attention\.output": {
#             "config": {
#                 "name": "lsqinteger",
#                 "level": 32,
#             }
#         },
#         "roberta\.encoder\.layer\.\d+\.output": {
#             "config": {
#                 "name": "lsqinteger",
#                 "level": 32,
#             }
#         },
#         "roberta\.encoder\.layer\.\d+\.intermediate": {
#             "config": {
#                 "name": "lsqinteger",
#                 "level": 32,
#             }
#         },
#         "classifier": {
#             "config": {
#                 "name": "lsqinteger",
#                 "level": 32,
#             }
#         },
#     }
#     mg, _ = quantize_module_transform_pass(bert, quan_pass_args)
#     # f = open(f"qann_model_arch.txt", "w")
#     # f.write(str(mg))
#     # f.close()

#     #
#     quan_pass_args = {
#         "by": "regex_name",
#         "roberta\.encoder\.layer\.\d+\.attention\.self": {
#             "config": {
#                 "name": "zip_tf",
#                 "level": 32,
#                 "neuron_type": "ST-BIF",
#             },
#         },
#     }
#     mg, _ = ann2snn_module_transform_pass(mg, quan_pass_args)
#     quan_pass_args = {
#         "by": "type",
#         "embedding": {
#             "config": {
#                 "name": "zip_tf",
#             },
#         },
#         "linear": {
#             "config": {
#                 "name": "unfold_bias",
#                 "level": 32,
#                 "neuron_type": "ST-BIF",
#             },
#         },
#         "conv2d": {
#             "config": {
#                 "name": "zip_tf",
#                 "level": 32,
#                 "neuron_type": "ST-BIF",
#             },
#         },
#         "layernorm": {
#             "config": {
#                 "name": "zip_tf",
#             },
#         },
#         "relu": {
#             "manual_instantiate": True,
#             "config": {
#                 "name": "identity",
#             },
#         },
#         "lsqinteger": {
#             "manual_instantiate": True,
#             "config": {
#                 "name": "st_bif",
#                 # Default values. These would be replaced by the values from the LSQInteger module, so it has no effect.
#                 # "q_threshold": 1,
#                 # "level": 32,
#                 # "sym": True,
#             },
#         },
#     }
#     mg, _ = ann2snn_module_transform_pass(mg, quan_pass_args)

#     # f = open(f"spiking_model_arch.txt", "w")
#     # f.write(str(mg))
#     # f.close()


# # import datasets as hf_datasets
# # sst2 = hf_datasets.load_dataset("gpt3mix/sst2")
# # train_df = sst2["train"]
# # dev_df = sst2["validation"]
# # test_df = sst2["test"]

# # max_seq_len = 50
# # epochs = 10
# # batch_size = 32
# # lr = 2e-5
# # patience = 5
# # max_grad_norm = 10
# # if_save_model = False
# # checkpoint = None

# test_ann2snn_module_transform_pass()
