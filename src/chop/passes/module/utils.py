import inspect
import re
import os
from copy import deepcopy
from typing import Tuple

import torch
from pathlib import Path
from functools import reduce
from transformers import PreTrainedModel, TFPreTrainedModel


def match_a_pattern(name: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.fullmatch(pattern, name)
        if match:
            return pattern
    return None


def check_is_huggingface_model(model):
    return isinstance(model, (PreTrainedModel, TFPreTrainedModel))
