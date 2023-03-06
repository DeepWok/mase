import types
import os
import functools

import torch

use_cuda = torch.cuda.is_available()
print("Using cuda:{}".format(use_cuda))
torch_cuda = torch.cuda if use_cuda else torch
device = torch.device("cuda" if use_cuda else "cpu")


def get_checkpoint_file(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            return file


def plt_model_load(model, checkpoint):
    state_dict = torch.load(checkpoint)["state_dict"]
    model.load_state_dict(state_dict)
    return model


def load_model(load_path, plt_model):
    if load_path is not None:
        if load_path.endswith(".ckpt"):
            checkpoint = load_path
        else:
            if load_path.endswith("/"):
                checkpoint = load_path + "best.ckpt"
            else:
                raise ValueError(
                    "if it is a directory, if must end with /; if it is a file, it must end with .ckpt"
                )
        plt_model = plt_model_load(plt_model, checkpoint)
        print(f"Loaded model from {checkpoint}")
    return plt_model
