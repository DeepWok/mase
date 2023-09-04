from os import PathLike
from .opt_patched import (
    OPTPatchedForCausalLM,
    OPTPatchedForSequenceClassification,
    OPTPatchedConfig,
    GPT2Tokenizer,
)
from ..utils import MaseModelInfo

# fmt: off
PATCHED_MODELS = {
    "facebook/opt-125m:patched": {
        "config_cls": OPTPatchedConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "sequence_classification": OPTPatchedForSequenceClassification,
        "causal_LM": OPTPatchedForCausalLM,
        "info": MaseModelInfo(
            model_source="patched", task_type="nlp", sequence_classification=True, causal_LM=True, fx_traceable=True
        ),
    },
    "facebook/opt-350m:patched": {
        "config_cls": OPTPatchedConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "sequence_classification": OPTPatchedForSequenceClassification,
        "causal_LM": OPTPatchedForCausalLM,
        "info": MaseModelInfo(
            model_source="patched", task_type="nlp", sequence_classification=True, causal_LM=True, fx_traceable=True
        ),
    },
    "facebook/opt-1.3b:patched": {
        "config_cls": OPTPatchedConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "sequence_classification": OPTPatchedForSequenceClassification,
        "causal_LM": OPTPatchedForCausalLM,
        "info": MaseModelInfo(
            model_source="patched", task_type="nlp", sequence_classification=True, causal_LM=True, fx_traceable=True
        ),
    },
    "facebook/opt-2.7b:patched": {
        "config_cls": OPTPatchedConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "sequence_classification": OPTPatchedForSequenceClassification,
        "causal_LM": OPTPatchedForCausalLM,
        "info": MaseModelInfo(
            model_source="patched", task_type="nlp", sequence_classification=True, causal_LM=True, fx_traceable=True
        ),
    },
    "facebook/opt-6.7b:patched": {
        "config_cls": OPTPatchedConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "sequence_classification": OPTPatchedForSequenceClassification,
        "causal_LM": OPTPatchedForCausalLM,
        "info": MaseModelInfo(
            model_source="patched", task_type="nlp", sequence_classification=True, causal_LM=True, fx_traceable=True
        ),
    },
}
# fmt: on


def is_patched_model(name: str):
    return name in PATCHED_MODELS


def get_patched_model_info(name: str) -> MaseModelInfo:
    if not is_patched_model(name):
        raise ValueError(f"Model `{name}` is not a patched model")
    return PATCHED_MODELS[name]["info"]


def get_patched_model(
    name: str,
    task: str,
    dataset_info: dict,
    pretrained: bool,
    checkpoint: str | PathLike,
):
    if not is_patched_model(name):
        raise ValueError(f"Model `{name}` is not a patched model")
    model_info = get_patched_model_info(name)
    hf_name = name.removesuffix(":patched")

    match task:
        case "cls" | "classification":
            assert (
                model_info.sequence_classification
            ), f"Model {name} does not support sequence classification"
            config = PATCHED_MODELS[name]["config_cls"].from_pretrained(
                hf_name, num_labels=dataset_info["num_classes"]
            )
            model_cls = PATCHED_MODELS[name]["sequence_classification"]
        case "lm" | "language_modeling":
            assert model_info.causal_LM, f"Model {name} does not support causal LM"
            config = PATCHED_MODELS[name]["config_cls"].from_pretrained(hf_name)
            model_cls = PATCHED_MODELS[name]["causal_LM"]
        case "tran" | "translation":
            assert model_info.seq2seqLM, f"Model {name} does not support seq2seq LM"
            config = PATCHED_MODELS[name]["config_cls"].from_pretrained(hf_name)
            model_cls = PATCHED_MODELS[name]["seq2seqLM"]
        case _:
            raise ValueError(f"Task {task} is not supported for {name}")

    if pretrained:
        if checkpoint is None:
            model = model_cls.from_pretrained(hf_name, config=config)
        else:
            model = model_cls.from_pretrained(checkpoint, config=config)
    else:
        model = model_cls(config)

    return model


def get_patched_model_tokenizer(name: str, checkpoint: str | PathLike = None):
    if not is_patched_model(name):
        raise ValueError(f"Model `{name}` is not a patched model")

    return PATCHED_MODELS[name]["tokenizer_cls"].from_pretrained(
        name.removesuffix(":patched") if checkpoint is None else checkpoint
    )


def get_patched_model_cls(name: str, task: str):
    if not is_patched_model(name):
        raise ValueError(f"Model `{name}` is not a patched model")

    model_info = get_patched_model_info(name)

    match task:
        case "cls" | "classification":
            assert (
                model_info.sequence_classification
            ), f"Model {name} does not support sequence classification"
            model_cls = PATCHED_MODELS[name]["sequence_classification"]
        case "lm" | "language_modeling":
            assert model_info.causal_LM, f"Model {name} does not support causal LM"
            model_cls = PATCHED_MODELS[name]["causal_LM"]
        case "tran" | "translation":
            assert model_info.seq2seqLM, f"Model {name} does not support seq2seq LM"
            model_cls = PATCHED_MODELS[name]["seq2seqLM"]
        case _:
            raise ValueError(f"Task {task} is not supported for {name}")

    return model_cls


def get_patched_model_config_cls(name: str):
    if not is_patched_model(name):
        raise ValueError(f"Model `{name}` is not a patched model")

    return PATCHED_MODELS[name]["config_cls"]


def get_patched_model_tokenizer_cls(name: str):
    if not is_patched_model(name):
        raise ValueError(f"Model `{name}` is not a patched model")

    return PATCHED_MODELS[name]["tokenizer_cls"]
