import logging
import os

import torch
import pdb

logger = logging.getLogger(__name__)


def load_lightning_ckpt_to_unwrapped_model(checkpoint: str, model: torch.nn.Module):
    """
    Load a PyTorch Lightning checkpoint to a PyTorch model.
    """
    #src_state_dict = torch.load(checkpoint)["state_dict"] # gpu
    src_state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))["state_dict"] #cpu
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


def load_unwrapped_ckpt(model_short_name, mask, is_quantize, checkpoint: str, model: torch.nn.Module):
    """
    Load a PyTorch state dict or checkpoint containing state dict to a PyTorch model.
    """
    print("has_quantized? : ",end="");print(is_quantize)
    state_dict = torch.load(checkpoint)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    if mask!=None:
        if "vgg" in model_short_name:
            state_dict['feature_layers.0.parametrizations.weight.0.mask'] = mask[0]
            state_dict['feature_layers.3.parametrizations.weight.0.mask'] = mask[1]
            state_dict['feature_layers.7.parametrizations.weight.0.mask'] = mask[2]
            state_dict['feature_layers.10.parametrizations.weight.0.mask'] = mask[3]
            state_dict['feature_layers.14.parametrizations.weight.0.mask'] = mask[4]
            state_dict['feature_layers.17.parametrizations.weight.0.mask'] = mask[5]
        elif "resnet" in model_short_name:
            state_dict['conv1.parametrizations.weight.0.mask'] = mask[0]
            state_dict['layer1.0.conv1.parametrizations.weight.0.mask'] = mask[1]
            state_dict['layer1.0.conv2.parametrizations.weight.0.mask'] = mask[2]
            state_dict['layer1.1.conv1.parametrizations.weight.0.mask'] = mask[3]
            state_dict['layer1.1.conv2.parametrizations.weight.0.mask'] = mask[4]
            state_dict['layer2.0.conv1.parametrizations.weight.0.mask'] = mask[5]
            state_dict['layer2.0.conv2.parametrizations.weight.0.mask'] = mask[6]
            state_dict['layer2.0.downsample.0.parametrizations.weight.0.mask'] = mask[7]
            state_dict['layer2.1.conv1.parametrizations.weight.0.mask'] = mask[8]
            state_dict['layer2.1.conv2.parametrizations.weight.0.mask'] = mask[9]
            state_dict['layer3.0.conv1.parametrizations.weight.0.mask'] = mask[10]
            state_dict['layer3.0.conv2.parametrizations.weight.0.mask'] = mask[11]
            state_dict['layer3.0.downsample.0.parametrizations.weight.0.mask'] = mask[12]
            state_dict['layer3.1.conv1.parametrizations.weight.0.mask'] = mask[13]
            state_dict['layer3.1.conv2.parametrizations.weight.0.mask'] = mask[14]
            state_dict['layer4.0.conv1.parametrizations.weight.0.mask'] = mask[15]
            state_dict['layer4.0.conv2.parametrizations.weight.0.mask'] = mask[16]
            state_dict['layer4.0.downsample.0.parametrizations.weight.0.mask'] = mask[17]
            state_dict['layer4.1.conv1.parametrizations.weight.0.mask'] = mask[18]
            state_dict['layer4.1.conv2.parametrizations.weight.0.mask'] = mask[19]
    
    if is_quantize == True:
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            new_key = new_key.replace('feature_layers.0.weight', 'feature_layers.0.parametrizations.weight.original') 
            new_key = new_key.replace('feature_layers.3.weight', 'feature_layers.3.parametrizations.weight.original') 
            new_key = new_key.replace('feature_layers.7.weight', 'feature_layers.7.parametrizations.weight.original') 
            new_key = new_key.replace('feature_layers.10.weight', 'feature_layers.10.parametrizations.weight.original') 
            new_key = new_key.replace('feature_layers.14.weight', 'feature_layers.14.parametrizations.weight.original') 
            new_key = new_key.replace('feature_layers.17.weight', 'feature_layers.17.parametrizations.weight.original') 
            new_state_dict[new_key] = value
        model.load_state_dict(state_dict=new_state_dict)
        return model
    
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
    model_short_name, mask, is_quantize, load_name: str, load_type: str = "mz", model: torch.nn.Module = None
    #load_name: str, load_type: str = "mz", model: torch.nn.Module = None
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
        model = load_unwrapped_ckpt(model_short_name, mask, is_quantize, checkpoint=load_name, model=model)
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
