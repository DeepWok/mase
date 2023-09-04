import os
from os import PathLike
from dataclasses import dataclass

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers.models.bert import (
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
)
from transformers.models.roberta import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
)
from transformers.models.opt import (
    OPTConfig,
    OPTForCausalLM,
    OPTForSequenceClassification,
)
from transformers.models.gpt2 import GPT2Tokenizer
from transformers.models.gpt_neo import (
    GPTNeoConfig,
    GPTNeoForCausalLM,
    GPTNeoForSequenceClassification,
)
from transformers.models.t5 import T5Config, T5Tokenizer, T5ForConditionalGeneration

from .utils import MaseModelInfo

# fmt: off
HF_NLP_MODELS = {
    "bert-base-uncased": {
        "config_cls": BertConfig,
        "tokenizer_cls": BertTokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True),
    },
    "bert-base-cased": {
        "config_cls": BertConfig,
        "tokenizer_cls": BertTokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True)
    },
    "bert-large-uncased": {
        "config_cls": BertConfig,
        "tokenizer_cls": BertTokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True)
    },
    "bert-large-cased": {
        "config_cls": BertConfig,
        "tokenizer_cls": BertTokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True)
    },
    "roberta-base": {
        "config_cls": RobertaConfig,
        "tokenizer_cls": RobertaTokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True)
    },
    "roberta-large": {
        "config_cls": RobertaConfig,
        "tokenizer_cls": RobertaTokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True)
    },
    "facebook/opt-125m": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True, causal_LM=True),
    },
    "facebook/opt-350m": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True, causal_LM=True),
    },
    "facebook/opt-1.3b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True, causal_LM=True),
    },
    "facebook/opt-2.7b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True, causal_LM=True),
    },
    "facebook/opt-6.7b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True, causal_LM=True),
    },
    "facebook/opt-13b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True, causal_LM=True),
    },
    "facebook/opt-30b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True, causal_LM=True),
    },
    "facebook/opt-66b": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True, causal_LM=True),
    },
    "EleutherAI/gpt-neo-125M": {
        "config_cls": GPTNeoConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True, causal_LM=True),
    },
    "EleutherAI/gpt-neo-1.3B": {
        "config_cls": GPTNeoConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True, causal_LM=True),
    },
    "EleutherAI/gpt-neo-2.7B": {
        "config_cls": GPTNeoConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True, causal_LM=True),
    },
    "EleutherAI/gpt-neox-20b": {
        "config_cls": GPTNeoConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", sequence_classification=True, causal_LM=True),
    },
    "t5-small": {
        "config_cls": T5Config,
        "tokenizer_cls": T5Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", seq2seqLM=True),
    },
    "t5-base": {
        "config_cls": T5Config,
        "tokenizer_cls": T5Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", seq2seqLM=True),
    },
    "t5-large": {
        "config_cls": T5Config,
        "tokenizer_cls": T5Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", seq2seqLM=True),
    },
    "google/flan-t5-small": {
        "config_cls": T5Config,
        "tokenizer_cls": T5Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", seq2seqLM=True),
    },
    "google/flan-t5-base": {
        "config_cls": T5Config,
        "tokenizer_cls": T5Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", seq2seqLM=True),
    },
    "google/flan-t5-large": {
        "config_cls": T5Config,
        "tokenizer_cls": T5Tokenizer,
        "info": MaseModelInfo(model_source="hf_transformers", task_type="nlp", seq2seqLM=True),
    },
}
# fmt: on


def is_hf_nlp_model(name: str) -> bool:
    return name in HF_NLP_MODELS


def get_hf_nlp_model_info(name: str) -> MaseModelInfo:
    if name not in HF_NLP_MODELS:
        raise ValueError(f"HuggingFace model {name} is not supported")
    return HF_NLP_MODELS[name]["info"]


def get_hf_nlp_model(
    name: str,
    task: str,
    dataset_info: dict,
    pretrained: bool,
    checkpoint: str = None,
):
    """
    Args:
        name: The name of the model.
        task: The task type.
        info: The dataset info.
        pretrained: Whether to load the pretrained model dict.
        checkpoint: The path to the checkpoint.

    ---
    A HuggingFace model checkpoint includes both config and model dict.
    - if `pretrained` is False, we will load the config from name/checkpoint and initialize the model randomly.
    - if `pretrained` is True, we will load the config and model dict from name/checkpoint.
    """
    if name not in HF_NLP_MODELS:
        raise ValueError(f"HuggingFace model {name} is not supported")

    model_info: MaseModelInfo = HF_NLP_MODELS[name]["info"]
    match task:
        case "lm" | "language_modeling":
            if not model_info.causal_LM:
                raise ValueError(f"Task {task} is not supported for {name}")

            if pretrained:
                model = AutoModelForCausalLM.from_pretrained(
                    name if checkpoint is None else checkpoint
                )
            else:
                config = AutoConfig.from_pretrained(
                    name if checkpoint is None else checkpoint
                )
                model = AutoModelForCausalLM.from_config(config)
        case "cls" | "classification":
            if not model_info.sequence_classification:
                raise ValueError(f"Task {task} is not supported for {name}")
            config = AutoConfig.from_pretrained(
                name if checkpoint is None else checkpoint,
                num_labels=dataset_info["num_classes"],
            )
            if pretrained:
                model = AutoModelForSequenceClassification.from_pretrained(
                    name if checkpoint is None else checkpoint, config=config
                )
            else:
                model = AutoModelForSequenceClassification.from_config(config)
        case "tran" | "translation":
            if not model_info.seq2seqLM:
                raise ValueError(f"Task {task} is not supported for {name}")
            if pretrained:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    name if checkpoint is None else checkpoint
                )
            else:
                config = AutoConfig.from_pretrained(
                    name if checkpoint is None else checkpoint
                )
                model = AutoModelForSeq2SeqLM.from_config(config)
        case _:
            raise ValueError(f"Task {task} is not supported for {name}")
    return model


def get_hf_nlp_model_tokenizer(name: str, checkpoint: str | PathLike = None):
    if name not in HF_NLP_MODELS:
        raise ValueError(f"HuggingFace model {name} is not supported")
    return AutoTokenizer.from_pretrained(name if checkpoint is None else checkpoint)


def get_hf_nlp_model_cls(name: str):
    raise NotImplementedError


def get_hf_nlp_model_config_cls(name: str):
    if name not in HF_NLP_MODELS:
        raise ValueError(f"HuggingFace model {name} is not supported")
    return HF_NLP_MODELS[name]["config_cls"]


def get_hf_nlp_model_tokenizer_cls(name: str):
    if name not in HF_NLP_MODELS:
        raise ValueError(f"HuggingFace model {name} is not supported")
    return HF_NLP_MODELS[name]["tokenizer_cls"]
