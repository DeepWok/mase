import pytest
import importlib

import transformers
from transformers import AutoModel
from transformers.utils.fx import _SUPPORTED_MODELS

from accelerate import init_empty_weights

from chop import MaseGraph
import chop.passes as passes


def create_masegraph(model_cls_name: str) -> MaseGraph:
    model_cls = getattr(transformers, model_cls_name)
    model_module_name = model_cls.__module__
    _CONFIG_FOR_DOC = importlib.import_module(model_module_name)._CONFIG_FOR_DOC
    config = getattr(transformers, _CONFIG_FOR_DOC)()

    with init_empty_weights():
        model = AutoModel.from_config(config)

    mg = MaseGraph(model)
    mg, _ = passes.init_metadata_analysis_pass(mg)
    return mg


for model_cls_name in _SUPPORTED_MODELS:

    def test_create_masegraph():
        mg = create_masegraph(model_cls_name)

    fn = test_create_masegraph
    fn.__name__ = f"test_create_masegraph_{model_cls_name}"

    globals()[fn.__name__] = fn
