import pytest
import transformers
from transformers import AutoModel
from transformers.utils.fx import _SUPPORTED_MODELS

from accelerate import init_empty_weights

from chop import MaseGraph
import chop.passes as passes


def create_masegraph(model_cls_name: str) -> MaseGraph:
    model_cls = getattr(transformers, model_cls_name)
    config = model_cls.config_class()

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
