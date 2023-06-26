import os
import torch
from .logger import logger


def check_when_to_load_and_how_to_load(
        action: str, is_pretrained: bool, load_name: str, load_type:str):

    to_train_to_test_or_to_test = action in ["train", "eval"]
    to_transform = action in ["transform"]
    if load_name is not None:
        if load_type is None:
            raise RuntimeError(
                "--load-type must be specified if --load is specified.")
        else:
            assert load_type in [
                "pt",
                "pl",
                "hf",
                "pkl",
            ], f"if --load is specified, --load-type should be one of ['pt', 'pl', 'hf', 'pkl'], but got {load_type}"

    if load_type == "hf" or is_pretrained:
        when_to_load = "init"
        how_to_load = "model_dependent"
    else:
        if to_transform:
            when_to_load = "transform"
            if load_name is None:
                when_to_load = how_to_load = None
            else:
                if load_type in ["pt", "pl"]:
                    how_to_load = load_type
                else:
                    raise RuntimeError(
                        f"transform only supports loading 'pt' and 'pl' checkpoint, but got --load-type={load_type}"
                    )
        elif to_train_to_test_or_to_test:
            when_to_load = "train_val_or_test"
            if load_name is None:
                when_to_load = how_to_load = None
            else:
                when_to_load = "train_val_or_test"
                if load_type in ["pt", "pl", "pkl"]:
                    how_to_load = load_type
                else:
                    raise RuntimeError(
                        f"train/test only supports 'pt', 'pl', and 'pkl' checkpoint, but got --load-type={load_type}"
                    )
        else:
            when_to_load = how_to_load = None
            if load_name is not None:
                logger.warn(
                    f"load_name {load_name} is provided but machop is not doing the modify-sw, train, validate-sw, or test-sw. The checkpoint will not be loaded"
                )

    # How to load will never be "hf"??
    if how_to_load == "hf":
        if is_pretrained and load_name:
            assert os.path.isdir(
                load_name
            ), f'''{load_name} is not a directory or does not exit. 
                To load local HuggingFace pretrained model, 
                --load-name should be a directory.'''
    elif how_to_load == "pt":
        assert load_name.endswith(".ckpt") or load_name.endswith(
            "pt"
        ), f'''To load PyTorch checkpoint, 
                --load-name should be a path to a .ckpt file, 
                but got {load_name}'''
        assert os.path.isfile(
            load_name
        ), f"the ckpt file {load_name} is not a file or does not exist"
    elif how_to_load == "pl":
        assert load_name.endswith(
            ".ckpt"
        ), f'''To load PyTorchLightning checkpoint, 
            --load-name should be a path to a .ckpt file, but got {load_name}'''
        assert os.path.isfile(
            load_name
        ), f"the ckpt file {load_name} is not a file or does not exist"
    elif how_to_load == "pkl":
        assert load_name.endswith(
            ".pkl"
        ), f'''To load pickle model,
            --load-name should a path to a .pkl file, but got {load_name}'''
        assert os.path.isfile(
            load_name
        ), f"the pkl file {load_name} is not a file or does not exist"

    return when_to_load, how_to_load


def load_pl_checkpoint_into_pt_model(checkpoint: str, model: torch.nn.Module):
    """
    The checkpoint can come from wrapped model or
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
    model.load_state_dict(new_tgt_state_dict)
    return model


def load_pt_model_into_pt_model(checkpoint: str, model: torch.nn.Module):
    state_dict = torch.load(checkpoint)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict=state_dict)
    return model


def load_pkl_model(checkpoint: str):
    with open(checkpoint, "rb") as f:
        model = pickle.load(f)
    return model


def load_model(
    load_name: str, load_type: str, model: torch.nn.Module = None
):
    assert load_type in [
        "pt",
        "pl",
        "pkl",
    ], f"load_type should be one of ['pt', 'pl', 'pkl']"

    if load_type == "pkl":
        model = load_pkl_model(load_name)
        logger.info(f"pkl model is loaded from {load_name}")
    elif load_type == "pl":
        model = load_pl_checkpoint_into_pt_model(load_name, model=model)
        logger.info(f"pl model is loaded from {load_name}")
    else:
        model = load_pt_model_into_pt_model(load_name, model=model)
        logger.info(f"pt model is loaded from {load_name}")
    return model
