import torch
import numpy

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
