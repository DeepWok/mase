import logging
import os

import torch

logger = logging.getLogger(__name__)


def load_lightning_ckpt_to_unwrapped_model(checkpoint: str, model: torch.nn.Module):
    """
    Load a PyTorch Lightning checkpoint to a PyTorch model.
    """
    src_state_dict = torch.load(checkpoint)["state_dict"]
    tgt_state_dict = model.state_dict()
    new_tgt_state_dict = {}
    for k, v in src_state_dict.items():
        if "model." in k:
            possible_tgt_k = ".".join(k.split(".")[1:])
        else:
            possible_tgt_k = k
        if possible_tgt_k in tgt_state_dict:
            new_tgt_state_dict[possible_tgt_k] = v
    model.load_state_dict(state_dict=new_tgt_state_dict)
    return model


def load_unwrapped_ckpt(checkpoint: str, model: torch.nn.Module):
    """
    Load a PyTorch state dict or checkpoint containing state dict to a PyTorch model.
    """
    state_dict = torch.load(checkpoint)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict=state_dict)
    return model


def load_graph_module_ckpt(checkpoint: str):
    """
    Load a serialized graph module.
    """
    if os.path.isdir(checkpoint):
        checkpoint = os.path.join(checkpoint, "graph_module.mz")
    model = torch.load(checkpoint)
    return model


def load_model(
    load_name: str, load_type: str = "mz", model: torch.nn.Module = None
) -> torch.nn.Module | torch.fx.GraphModule:
    """Load a pytorch/lightning/mase checkpoint to a model.

    Args:
        load_name (str): path to the checkpoint
        load_type (str, optional): checkpoint type, must be one of ['pt', 'pl', 'mz'],
        representing pytorch/lightning/mase. Defaults to "auto" inferred from the extension.
        model (torch.nn.Module, optional): Model candidate to load checkpoint.
        Note that 'ms' checkpoint loads the model as well as state dict, thus does not need this arg. Defaults to None.

    Raises:
        ValueError: Unknown extension for 'load_type'.

    Returns:
        nn.Module/fx.GraphModule: the model with the checkpoint loaded
    """
    if load_type == "hf":
        raise RuntimeError(
            "HuggingFace checkpoint should be loaded using model_inst_fn."
        )
    elif load_type not in ["pt", "pl", "mz"]:
        raise ValueError(f"Unknown extension for 'load_type': {load_type}")

    if load_type == "pt":
        model = load_unwrapped_ckpt(checkpoint=load_name, model=model)
        logger.info(f"Loaded pytorch checkpoint from {load_name}")
    elif load_type == "pl":
        model = load_lightning_ckpt_to_unwrapped_model(
            checkpoint=load_name, model=model
        )
        logger.info(f"Loaded pytorch lightning checkpoint from {load_name}")
    else:
        model = load_graph_module_ckpt(checkpoint=load_name)
        logger.info(f"Loaded mase checkpoint from {load_name}")
    return model
