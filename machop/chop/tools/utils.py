import numpy as np
import os
import pickle

import colorlog
import torch
import subprocess

from torch import Tensor

# LUTNet
import itertools

use_cuda = torch.cuda.is_available()
torch_cuda = torch.cuda if use_cuda else torch
device = torch.device("cuda:0" if use_cuda else "cpu")


def to_numpy(x):
    if use_cuda:
        x = x.cpu()
    return x.detach().numpy()


def to_tensor(x):
    return torch.from_numpy(x).to(device)


def copy_weights(src_weight: Tensor, tgt_weight: Tensor):
    with torch.no_grad():
        tgt_weight.copy_(src_weight)


def get_checkpoint_file(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            return file


def execute_cli(cmd, log_output: bool = True, log_file=None, cwd="."):
    if log_output:
        logger.debug("{} (cwd = {})".format(subprocess.list2cmdline(cmd), cwd))
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, cwd=cwd
        ) as result:
            if log_file:
                f = open(log_file, "w")
            if result.stdout or result.stderr:
                logger.info("")
            if result.stdout:
                for line in result.stdout:
                    if log_file:
                        f.write(line)
                    line = line.rstrip("\n")
                    logging.trace(line)
            if result.stderr:
                for line in result.stderr:
                    if log_file:
                        f.write(line)
                    line = line.rstrip("\n")
                    logging.trace(line)
            if log_file:
                f.close()
    else:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, cwd=cwd)
    return result.returncode


def get_factors(n):
    factors = np.sort(
        list(
            set(
                functools.reduce(
                    list.__add__,
                    ([i, n // i] for i in range(1, int(n**0.5) + 1) if n % i == 0),
                )
            )
        )
    )
    factors = [int(x) for x in factors]
    return factors


# ---------------------------------------------
# LUTNet helpers
# ---------------------------------------------
def generate_truth_table(k: int, tables_count: int, device: None) -> torch.Tensor:
    """This function generate truth tables with size of k * (2**k) * tables_count

    Args:
        k (int): truth table power
        tables_count (int): number of truth table repetition
        device (str): target device of the result

    Returns:
        torch.Tensor: 2d torch tensor with k*tables_count rows and (2**k) columns
    """

    table = torch.from_numpy(np.array(list(itertools.product([-1, 1], repeat=k)))).T
    return torch.vstack([table] * tables_count).to(device)


def init_LinearLUT_weight(
    levels,
    k,
    original_pruning_mask,
    original_weight,
    in_features,
    out_features,
    new_module,
):
    # Initialize the weight based on the trained binaried network
    # weight shape of the lagrange trainer [tables_count, self.kk]
    input_mask = new_module.input_mask.reshape(
        -1, k * in_features
    )  # (out_feature, k * in_feature)

    expanded_original_weight = original_weight[
        np.arange(out_features)[:, np.newaxis], input_mask
    ].reshape(-1, k, 1)
    index_weight, reconnected_weight = (
        expanded_original_weight[:, 0, :],
        expanded_original_weight[:, 1:, :],
    )  # [input_feature * output_feature, 1]

    # Establish pruning mask
    expanded_pruning_masks = original_pruning_mask[
        np.arange(out_features)[:, np.newaxis], input_mask
    ].reshape(
        -1, k, 1
    )  # (out_feature * in_feature, k, 1)

    pruned_connection = expanded_pruning_masks[:, 0, :]

    d = generate_truth_table(k=k, tables_count=1, device=None)
    initialized_weight = index_weight * d[0, :]
    for extra_input_index in range(1, k):
        pruned_extra_input = ~(
            expanded_pruning_masks[:, extra_input_index, :].squeeze().bool()
        )

        initialized_weight[pruned_extra_input, :] = (
            initialized_weight[pruned_extra_input, :]
            + (reconnected_weight * d[extra_input_index, :]).squeeze()[
                pruned_extra_input, :
            ]
        )

    initialized_weight = torch.cat([initialized_weight] * levels, dim=0)
    pruned_connection = torch.cat([pruned_connection] * levels, dim=0)
    return initialized_weight, pruned_connection


def init_Conv2dLUT_weight(
    levels,
    k,
    original_pruning_mask,
    original_weight,
    out_channels,
    in_channels,
    kernel_size,
    new_module,
):
    # Initialize the weight based on the trained binaried network
    # weight shape of the lagrange trainer [tables_count, self.kk]
    input_mask = new_module.input_mask.reshape(
        -1,
        in_channels * kernel_size[0] * kernel_size[1] * k,
        3,
    )  # [oc, k * kh * kw * ic ,3[ic,kh,kw]]
    expanded_original_weight = original_weight[
        np.arange(out_channels)[:, np.newaxis],
        input_mask[:, :, 0],
        input_mask[:, :, 1],
        input_mask[:, :, 2],
    ].reshape(
        -1, k, 1
    )  # [oc * ic * kw * kh , k, 1]
    index_weight, reconnected_weight = (
        expanded_original_weight[:, 0, :],
        expanded_original_weight[:, 1:, :],
    )

    # Establish pruning mask
    expanded_pruning_masks = original_pruning_mask[
        np.arange(out_channels)[:, np.newaxis],
        input_mask[:, :, 0],
        input_mask[:, :, 1],
        input_mask[:, :, 2],
    ].reshape(
        -1, k, 1
    )  # (out_feature * in_feature, k, 1)
    pruned_connection = expanded_pruning_masks[
        :, 0, :
    ]  # [input_feature * output_feature, 1]

    d = generate_truth_table(k=k, tables_count=1, device=None)
    initialized_weight = index_weight * d[0, :]
    for extra_input_index in range(1, k):
        pruned_extra_input = ~(
            expanded_pruning_masks[:, extra_input_index, :].squeeze().bool()
        )

        initialized_weight[pruned_extra_input, :] = (
            initialized_weight[pruned_extra_input, :]
            + (reconnected_weight * d[extra_input_index, :]).squeeze()[
                pruned_extra_input, :
            ]
        )

    initialized_weight = torch.cat([initialized_weight] * levels, dim=0)
    pruned_connection = torch.cat([pruned_connection] * levels, dim=0)
    return initialized_weight, pruned_connection
