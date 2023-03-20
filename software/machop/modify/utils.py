import numpy
import torch
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


def plt_model_load(model, checkpoint):
    state_dict = torch.load(checkpoint)["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if "model." in k:
            # import pdb; pdb.set_trace()
            new_state_dict[".".join(k.split(".")[1:])] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    return model


def load_checkpoint_into_model(checkpoint: str, model: torch.nn.Module):
    """
    The checkpoint can come from wrapped model or
    """
    assert checkpoint.endswith(
        ".ckpt"
    ), f"Checkpoint should be a .ckpt file, but get {checkpoint}"
    src_state_dict = torch.load(checkpoint)["state_dict"]
    tgt_state_dict = model.state_dict()
    new_tgt_state_dict = {}
    for k, v in src_state_dict.items():
        if "model." in k:
            possible_tgt_k = ".".join(k.split(".")[1:])
        else:
            possible_tgt_k = k
        if possible_tgt_k in tgt_state_dict:
            new_tgt_state_dict[k] = v
    model.load_state_dict(new_tgt_state_dict)
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


def copy_weights(src_weight: Tensor, tgt_weight: Tensor):
    with torch.no_grad():
        tgt_weight.copy_(src_weight)
