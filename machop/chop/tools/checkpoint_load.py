import logging
import os
from chop.passes.graph.transforms.pruning.sparse_parameterization import (
    FakeSparseWeight,
)
from chop.passes.graph.transforms.pruning.prune import activation_pruning_pass
import torch
import torch as nn

logger = logging.getLogger(__name__)


def load_lightning_ckpt_to_unwrapped_model(checkpoint: str, model: torch.nn.Module):
    """
    Load a PyTorch Lightning checkpoint to a PyTorch model.
    """
    src_state_dict = torch.load(checkpoint)["state_dict"]
    tgt_state_dict = model.state_dict()
    new_tgt_state_dict = {}
    src_state_dict = return_unparametrised_state_dict(src_state_dict, tgt_state_dict)
    for k, v in src_state_dict.items():
        if "model." in k:
            possible_tgt_k = ".".join(k.split(".")[1:])
        else:
            possible_tgt_k = k
        if possible_tgt_k in tgt_state_dict:
            new_tgt_state_dict[possible_tgt_k] = v

    model.load_state_dict(state_dict=new_tgt_state_dict)
    return model


def return_unparametrised_state_dict(param_state_dict, model_state_dict):
    new_tgt_state_dict = {}
    for k, v in param_state_dict.items():
        if "model." in k:
            possible_tgt_k = ".".join(k.split(".")[1:])
        else:
            possible_tgt_k = k
        new_tgt_state_dict[possible_tgt_k] = v
    new_unparam_state_dict = {}

    for k, v in new_tgt_state_dict.items():
        # Check if the key contains the specific substring and modify it accordingly
        if "parametrizations.weight" in k:
            # Create a new key by removing the unwanted part
            possible_tgt_k = k.replace(".parametrizations.weight.original", ".weight")
        else:
            possible_tgt_k = k

        # Update the temporary dictionary with the new key (if modified) or the old key
        if possible_tgt_k in model_state_dict:
            new_unparam_state_dict[possible_tgt_k] = v
    if len(new_unparam_state_dict) == 0:
        new_unparam_state_dict = model_state_dict

    return new_unparam_state_dict


def load_unwrapped_ckpt(checkpoint: str, model: torch.nn.Module):
    """
    Load a PyTorch state dict or checkpoint containing state dict to a PyTorch model.
    """
    state_dict = torch.load(checkpoint)

    # Write code to separate_state_dict
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]

    # reapply_parametrizations_from_state_dict(model, state_dict)
    state_dict = return_unparametrised_state_dict(state_dict, model.state_dict())
    # model.load_state_dict(loaded_state_dict)
    model.load_state_dict(state_dict=state_dict)
    return model


def reappply_activations(graph, state_dict_list):
    for activation_config in state_dict_list:
        activation_config = {"activation": activation_config}
        graph = activation_pruning_pass(graph, activation_config)
    return graph


def reapply_parametrizations_from_state_dict(model, state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if "model." in k:
            possible_tgt_k = ".".join(k.split(".")[1:])
        else:
            possible_tgt_k = k
        new_state_dict[possible_tgt_k] = v
    for key, tensor in new_state_dict.items():
        # Identify mask entries based on a naming convention or pattern
        if "mask" in key:

            parts = key.split(".")

            try:
                seq_index = int(parts[1])
                layer = model.seq_blocks[seq_index]

            except:
                try:
                    module_path_parts = parts[:-4]  # Remove the last four parts
                    # module_name = '.'.join(module_path_parts)
                    block = getattr(model, parts[0])
                    layer = block[int(parts[1])]
                except:
                    layer = getattr(model, parts[0])

            # Determine the parameter name that the mask is associated with
            param_name = "weight"

            # Directly use the tensor as the mask
            device = next(model.parameters()).device
            mask = tensor.to(device)

            # Register the new mask parametrization
            torch.nn.utils.parametrize.register_parametrization(
                layer, param_name, FakeSparseWeight(mask)
            )


def reapply_parametrizations_mg_module(graph, state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if "model." in k:
            possible_tgt_k = ".".join(k.split(".")[1:])
        else:
            possible_tgt_k = k
        new_state_dict[possible_tgt_k] = v
    for key, tensor in new_state_dict.items():
        # Identify mask entries based on a naming convention or pattern
        if "mask" in key:
            parts = key.split(".")
            module_path_parts = parts[:-4]  # Remove the last four parts
            module_name = ".".join(module_path_parts)

            # Determine the parameter name that the mask is associated with
            param_name = module_name
            for node in graph.fx_graph.nodes:
                if node.op == "call_module":
                    name = node.target
                    if name == param_name:
                        # Directly use the tensor as the mask
                        device = next(graph.model.parameters()).device
                        mask = tensor.to(device)
                        # Register the new mask parametrization
                        torch.nn.utils.parametrize.register_parametrization(
                            graph.modules[param_name], "weight", FakeSparseWeight(mask)
                        )

    return graph


def load_state_dict(load_name: str, load_type: str):
    if load_type == "pt":
        state_dict = torch.load(load_name)
        logger.info(f"Loaded pytorch checkpoint from {load_name}")
    elif load_type == "pl":
        if not load_name.endswith(".ckpt"):
            logger.warning(
                f"Lightning checkpoint should end with '.ckpt', but got {load_name}"
            )
        state_dict = torch.load(load_name)["state_dict"]
        logger.info(f"Loaded pytorch lightning checkpoint from {load_name}")
    else:
        logger.info(f"!! Not supported")

    return state_dict


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
        if not load_name.endswith(".ckpt"):
            logger.warning(
                f"Lightning checkpoint should end with '.ckpt', but got {load_name}"
            )
        model = load_lightning_ckpt_to_unwrapped_model(
            checkpoint=load_name, model=model
        )
        logger.info(f"Loaded pytorch lightning checkpoint from {load_name}")
    else:
        assert load_name.endswith(
            ".mz"
        ), f"Invalid extension for 'load_type=mz': {load_name}, must be a '.mz' file, but got {load_name}."

        model = load_graph_module_ckpt(checkpoint=load_name)
        logger.info(f"Loaded mase checkpoint from {load_name}")
    return model
