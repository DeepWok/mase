"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 01:55:29
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 01:55:30
"""

import os
import argparse
import json
import logging
import logging.handlers
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch


__all__ = [
    "ensure_dir",
    "read_json",
    "write_json",
    "profile",
    "print_stat",
    "Timer",
    "TimerCtx",
    "TorchTracemalloc",
    "fullprint",
    "setup_default_logging",
    "Logger",
    "logger",
    "get_logger",
    "ArgParser",
    "disable_tf_warning",
    "AverageMeter",
]


def ensure_dir(dirname, exist_ok: bool = True):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=exist_ok)


def read_json(fname):
    with open(fname, "rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with open(fname, "wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def profile(func=None, timer=True):
    from functools import wraps, partial
    import time

    if func == None:
        return partial(profile, timer=timer)

    @wraps(func)
    def wrapper(*args, **kw):
        if timer:
            local_time = time.time()
            res = func(*args, **kw)
            end_time = time.time()
            print("[I] <%s> runtime: %.3f ms" % (func.__name__, (end_time - local_time) * 1000))
        else:
            res = func(*args, **kw)
        return res

    return wrapper


def print_stat(x, message="", verbose=True):
    if verbose:
        if isinstance(x, torch.Tensor):
            if torch.is_complex(x):
                x = torch.view_as_real(x)
            print(
                message + f"min = {x.data.min().item():-15f} max = {x.data.max().item():-15f} mean = {x.data.mean().item():-15f} std = {x.data.std().item():-15f}"
            )
        elif isinstance(x, np.ndarray):
            print(
                message + f"min = {np.min(x):-15f} max = {np.max(x):-15f} mean = {np.mean(x):-15f} std = {np.std(x):-15f}"
            )


class Timer(object):
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


class TimerCtx:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


class TorchTracemalloc(object):
    def __init__(self, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose

    def __enter__(self):
        self.begin = self._b2mb(torch.cuda.memory_allocated())
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        return self

    def _b2mb(self, x):
        return x / 2 ** 20

    def __exit__(self, *exc):
        self.end = self._b2mb(torch.cuda.memory_allocated())
        self.peak = self._b2mb(torch.cuda.max_memory_allocated())
        self.used = self.end - self.begin
        self.peaked = self.peak - self.begin
        if self.verbose:
            print(f"Delta used/peaked {self.used:.2f} MB / {self.peaked:.2f} MB")
            print(f"Current used/peaked {self.end:.2f} MB / {self.peak:.2f} MB")


class fullprint:
    "context manager for printing full numpy arrays"

    def __init__(self, **kwargs):
        """linewidth=75; precision=8"""
        kwargs.setdefault("threshold", np.inf)
        self.opt = kwargs

    def __enter__(self):
        self._opt = np.get_printoptions()
        np.set_printoptions(**self.opt)

    def __exit__(self, type, value, traceback):
        np.set_printoptions(**self._opt)


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;21m"
    reset = "\x1b[0m"
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_default_logging(default_level=logging.INFO, default_file_level=logging.INFO, log_path=""):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(default_file_level)
        logging.root.addHandler(file_handler)


class Logger(object):
    def __init__(self, console=True, logfile=None, console_level=logging.INFO, logfile_level=logging.INFO):
        super().__init__()
        self.logfile = logfile
        self.console_level = console_level
        self.logifle_level = logfile_level
        assert (
            console == True or logfile is not None
        ), "At least enable one from console or logfile for Logger"
        # 第一步，创建一个logger
        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(logging.INFO)  # Log等级总开关
        self.logger.propagate = False

        # formatter = logging.Formatter(
        # "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        formatter = CustomFormatter()

        # 第三步，再创建一个handler，用于输出到控制台
        if console:
            ch = logging.StreamHandler()
            ch.setLevel(self.console_level)  # 输出到console的log等级的开关
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        if self.logfile is not None:
            fh = logging.FileHandler(self.logfile, mode="w")
            fh.setLevel(self.logifle_level)  # 输出到file的log等级的开关
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


def get_logger(name="default", default_level=logging.INFO, default_file_level=logging.INFO, log_path=""):
    setup_default_logging(
        default_level=default_level, default_file_level=default_file_level, log_path=log_path
    )
    return logging.getLogger(name)


logger = get_logger()


class ArgParser(object):
    def __init__(self, load_json=None, save_json=None):
        super().__init__()
        self.load_json = load_json
        self.save_json = save_json
        self.args = None
        self.parser = argparse.ArgumentParser("Argument Parser")

    def add_arg(self, *args, **keywords):
        self.parser.add_argument(*args, **keywords)

    def parse_args(self):
        if self.load_json is not None:
            assert os.path.exists(self.load_json), logging.error(
                f"Configuration JSON {self.load_json} not found"
            )
            json = read_json(self.load_json)
            t_args = argparse.Namespace()
            t_args.__dict__.update(json)
            self.args = self.parser.parse_args(args=[], namespace=t_args)
        else:
            self.args = self.parser.parse_args()
        return self.args

    def print_args(self):
        # Print arguments to std out
        # and save argument values to yaml file
        print("Arguments:")
        for p in vars(self.args).items():
            print(f"\t{p[0]:30}{str(p[1]):20}")
        print("\n")

    def dump_args(self, json_file=None):
        if json_file is None:
            if self.save_json is None:
                logging.error("Skip dump configuration JSON. Please specify json_file")
                return False
            else:
                ensure_dir(os.path.dirname(self.save_json))
                logging.warning(f"Dump to the initialized JSON file {self.save_json}")
                write_json(vars(self.args), self.save_json)
        else:
            ensure_dir(os.path.dirname(json_file))
            logging.info(f"Dump to JSON file {json_file}")
            write_json(vars(self.args), json_file)
        # with open(self.file, 'w') as f:
        #     yaml.dump(vars(self.args), f, default_flow_style=False)
        #     print(f"[I] Arguments dumped to {file}")


def disable_tf_warning():
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    import tensorflow as tf

    if hasattr(tf, "contrib") and type(tf.contrib) != type(tf):
        tf.contrib._warning = None
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # tf.logging.set_verbosity(tf.logging.ERROR)

    import logging

    logging.getLogger("tensorflow").setLevel(logging.ERROR)


class Meter(object):
    """Base class for Meters."""

    def __init__(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def reset(self):
        raise NotImplementedError

    @property
    def smoothed_value(self) -> float:
        """Smoothed value used for logging."""
        raise NotImplementedError


def safe_round(number, ndigits):
    if hasattr(number, "__round__"):
        return round(number, ndigits)
    elif torch is not None and torch.is_tensor(number) and number.numel() == 1:
        return safe_round(number.item(), ndigits)
    elif np is not None and np.ndim(number) == 0 and hasattr(number, "item"):
        return safe_round(number.item(), ndigits)
    else:
        return number


def type_as(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.to(b)
    else:
        return a


class AverageMeter(Meter):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f", round: Optional[int] = None) -> None:
        self.name = name
        self.fmt = fmt
        self.round = round
        self.reset()

    def reset(self):
        self.val = None  # most recent update
        self.sum = 0  # sum from all updates
        self.count = 0  # total n from all updates
        self.avg = 0

    def update(self, val, n=1):
        if val is not None:
            self.val = val
            if n > 0:
                self.sum = type_as(self.sum, val) + (val * n)
                self.count = type_as(self.count, n) + n
        self.avg = self.sum / self.count if self.count > 0 else self.val

    def state_dict(self):
        return {
            "val": self.val,
            "sum": self.sum,
            "count": self.count,
            "round": self.round,
        }

    def load_state_dict(self, state_dict):
        self.val = state_dict["val"]
        self.sum = state_dict["sum"]
        self.count = state_dict["count"]
        self.round = state_dict.get("round", None)

    @property
    def smoothed_value(self) -> float:
        val = self.avg
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
