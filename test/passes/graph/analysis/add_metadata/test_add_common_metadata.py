import torch
import pytest
import importlib

import transformers
from transformers import AutoModel
from transformers.utils.fx import _SUPPORTED_MODELS

from accelerate import init_empty_weights

from chop import MaseGraph
import chop.passes as passes


def add_common_metadata(model_cls_name: str) -> MaseGraph:
    model_cls = getattr(transformers, model_cls_name)
    model_module_name = model_cls.__module__
    _CONFIG_FOR_DOC = importlib.import_module(model_module_name)._CONFIG_FOR_DOC
    config = getattr(transformers, _CONFIG_FOR_DOC)()

    # with init_empty_weights():
    model = AutoModel.from_config(config)

    mg = MaseGraph(model)
    mg, _ = passes.init_metadata_analysis_pass(mg)

    # mg.fx_graph.print_tabular()

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (
            1,
            128,
            config.hidden_size,
        ),
        device="meta",
    )
    mg, _ = passes.add_common_metadata_analysis_pass(
        mg,
        pass_args={
            "dummy_in": {
                "input_ids": input_ids,
            },
        },
    )
    return mg


UNSUPPORTED = [
    "AlbertForMaskedLM",
    "AlbertForMultipleChoice",
    "AlbertForSequenceClassification" "AlbertForQuestionAnswering",
    "AlbertForPreTraining",
    "AlbertForTokenClassification",
    "ElectraModel",
    "ElectraForMaskedLM",
    "ElectraForMultipleChoice",
    "ElectraForCausalLM",
    "ElectraForSequenceClassification",
    "Speech2TextForConditionalGeneration",
    "ElectraForTokenClassification" "Speech2TextModel",
    "DonutSwinModel",
    "Speech2Text2Decoder",
    "Speech2Text2ForCausalLM",
    "GPT2DoubleHeadsModel",
    "GPT2ForQuestionAnswering",
    "GPT2ForSequenceClassification",
    "GPT2ForTokenClassification",
    "ViTForImageClassification",
    "ViTForMaskedImageModeling",
    "ViTModel",
    "Wav2Vec2ForCTC",
    "XGLMForCausalLM",
]

# TODO: debug this
# mg = add_common_metadata("OPTModel")
# for model_cls_name in _SUPPORTED_MODELS:
#     if model_cls_name in UNSUPPORTED:
#         continue
#     try:
#         def test_add_common_metadata():
#             mg = add_common_metadata(model_cls_name)

#         fn = test_add_common_metadata
#         fn.__name__ = f"test_add_common_metadata_{model_cls_name}"
#         globals()[fn.__name__] = fn
#     except:
#         failed.append(model_cls_name)
# print(failed)
