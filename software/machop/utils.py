import functools
import logging
import os
import pickle

import colorlog
import torch

# use_cuda = torch.cuda.is_available()
# print("Using cuda:{}".format(use_cuda))
# torch_cuda = torch.cuda if use_cuda else torch
# device = torch.device("cuda" if use_cuda else "cpu")


# -------------------------------
# MASE Logger
# -------------------------------


def getLogger(name: str, logFile: str = "", console: bool = True) -> logging.Logger:
    # add a trace level
    logging.TRACE = logging.DEBUG - 5
    logging.addLevelName(logging.TRACE, "TRACE")
    logging.Logger.trace = functools.partialmethod(logging.Logger.log, logging.TRACE)
    logging.trace = functools.partial(logging.log, logging.TRACE)

    logger = logging.getLogger(name)
    logger.setLevel(logging.TRACE)

    if logFile:
        if os.path.isfile(logFile):
            os.remove(logFile)

        # File handle
        class customFileFormat(logging.Formatter):
            format = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"

            def format(self, record):
                logformat = (
                    "%(message)s"
                    if record.levelno == logging.TRACE
                    else "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
                )
                formatter = logging.Formatter(logformat, "%Y-%m-%d %H:%M:%S")
                return formatter.format(record)

        fh = logging.FileHandler(logFile)
        fh.setFormatter(customFileFormat())
        fh.setLevel(logging.TRACE)
        logger.addHandler(fh)

    # Console handler
    if console:
        ch = logging.StreamHandler()

        class customConsoleFormat(logging.Formatter):
            format = "[%(asctime)s][%(name)s][%(levelname)s] %(message)s"

            def format(self, record):
                traceformat = logging.Formatter("%(message)s", "%Y-%m-%d %H:%M:%S")
                colorformat = colorlog.ColoredFormatter(
                    "%(log_color)s[%(asctime)s][%(name)s][%(levelname)s]%(reset)s"
                    + " %(message_log_color)s%(message)s",
                    "%Y-%m-%d %H:%M:%S",
                    log_colors={
                        "DEBUG": "cyan",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "red,bg_white",
                    },
                    secondary_log_colors={
                        "message": {"ERROR": "red", "CRITICAL": "red"}
                    },
                )
                logformat = (
                    traceformat if record.levelno == logging.TRACE else colorformat
                )
                return logformat.format(record)

        ch.setFormatter(customConsoleFormat())
        ch.setLevel(logging.TRACE)
        logger.addHandler(ch)
    return logger


logger = getLogger(__name__)


def get_checkpoint_file(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            return file


def check_when_to_load_and_how_to_load(
    to_modify_sw: bool,
    to_train: bool,
    to_validate_sw: bool,
    to_test_sw: bool,
    is_pretrained: bool,
    load_name: str,
    load_type: str,
):
    to_train_to_test_or_to_test = to_train or to_validate_sw or to_test_sw
    when_to_load = how_to_load = None
    assert not (
        to_train_to_test_or_to_test and to_modify_sw
    ), "--modify-sw and --train/--validate-sw/--test-sw cannot both be True. Please run these two commands sequentially."
    if load_name is not None:
        if load_type is None:
            raise RuntimeError("--load-type must be specified if --load is specified.")
        else:
            assert load_type in [
                "pt",
                "pl",
                "hf",
                "pkl",
            ], f"if --load is specified, --load-type should be one of ['pt', 'pl', 'hf', 'pkl'], but got {load_type}"

    if load_type == "hf" or is_pretrained:
        if is_pretrained and load_type != "hf":
            logger.warning(
                f"--pretrained is specified, thus --load-type=hf is expected, but got --load-type={load_type}"
            )
        when_to_load = "init"
        how_to_load = "hf"
    else:
        if to_modify_sw:
            when_to_load = "modify-sw"
            if load_name is None:
                when_to_load = how_to_load = None
            else:
                if load_type == "pt":
                    how_to_load = "pt"
                elif load_type == "pl":
                    how_to_load = "pl"
                else:
                    raise RuntimeError(
                        f"modify-sw only supports loading 'pt' and 'pl' checkpoint, but got --load-type={load_type}"
                    )
        elif to_train_to_test_or_to_test:
            when_to_load = "train_val_or_test"
            if load_name is None:
                when_to_load = how_to_load = None
            else:
                when_to_load = "train_val_or_test"
                if load_type == "pt":
                    how_to_load = "pt"
                elif load_type == "pl":
                    how_to_load = "pl"
                elif load_type == "pkl":
                    how_to_load = "pkl"
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

    if how_to_load == "hf":
        if is_pretrained and load_name:
            assert os.path.isdir(
                load_name
            ), f"{load_name} is not a directory or does not exit. To load local HuggingFace pretrained model, --load-name should be a directory."
    elif how_to_load == "pt":
        assert load_name.endswith(".ckpt") or load_name.endswith(
            "pt"
        ), f"To load PyTorch checkpoint, --load-name should be a path to a .ckpt file, but got {load_name}"
        assert os.path.isfile(
            load_name
        ), f"the ckpt file {load_name} is not a file or does not exist"
    elif how_to_load == "pl":
        assert load_name.endswith(
            ".ckpt"
        ), f"To load PyTorchLightning checkpoint, --load-name should be a path to a .ckpt file, but got {load_name}"
        assert os.path.isfile(
            load_name
        ), f"the ckpt file {load_name} is not a file or does not exist"
    elif how_to_load == "pkl":
        assert load_name.endswith(
            ".pkl"
        ), f"To load pickle model, --load-name should a path to a .pkl file, but got {load_name}"
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


def load_pt_pl_or_pkl_checkpoint_into_pt_model(
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
