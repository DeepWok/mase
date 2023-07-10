import numpy as np
import os
import pickle

import colorlog
import torch
import subprocess

from torch import Tensor

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
