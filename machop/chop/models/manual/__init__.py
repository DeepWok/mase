from os import PathLike
import logging
from ..utils import MaseModelInfo
from .bert_quantized import (
    BertQuantizedConfig,
    BertQuantizedForSequenceClassification,
    BertTokenizer,
)
from .llama_plain import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaTokenizer,
)
from .llama_quantized import (
    LlamaQuantizedConfig,
    LlamaQuantizedForCausalLM,
    LlamaQuantizedForSequenceClassification,
)
from .opt_plain import (
    OPTConfig,
    OPTForCausalLM,
    OPTForSequenceClassification,
    GPT2Tokenizer,
)
from .opt_quantized import (
    OPTQuantizedConfig,
    OPTQuantizedForCausalLM,
    OPTQuantizedForSequenceClassification,
)

logger = logging.getLogger(__name__)

# fmt: off
MANUAL_MODELS = {
    "bert_quantized": {
        "config_cls": BertQuantizedConfig,
        "tokenizer_cls": BertTokenizer,
        "info": MaseModelInfo(model_source="manual", task_type="nlp", sequence_classification=True),
        "sequence_classification": BertQuantizedForSequenceClassification,
    },
    "llama_plain": {
        "config_cls": LlamaConfig,
        "tokenizer_cls": LlamaTokenizer,
        "info": MaseModelInfo(model_source="manual", task_type="nlp", sequence_classification=True, causal_LM=True),
        "sequence_classification": LlamaForSequenceClassification,
        "causal_LM": LlamaForCausalLM,
    },
    "llama_quantized": {
        "config_cls": LlamaQuantizedConfig,
        "tokenizer_cls": LlamaTokenizer,
        "info": MaseModelInfo(model_source="manual", task_type="nlp", sequence_classification=True, causal_LM=True),
        "sequence_classification": LlamaQuantizedForSequenceClassification,
        "causal_LM": LlamaQuantizedForCausalLM,
    },
    "opt_plain": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": MaseModelInfo(model_source="manual", task_type="nlp", sequence_classification=True, causal_LM=True),
        "sequence_classification": OPTForSequenceClassification,
        "causal_LM": OPTForCausalLM,
    },
    "opt_quantized": {
        "config_cls": OPTQuantizedConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": MaseModelInfo(model_source="manual", task_type="nlp", sequence_classification=True, causal_LM=True),
        "sequence_classification": OPTQuantizedForSequenceClassification,
        "causal_LM": OPTQuantizedForCausalLM,
    },
}
# fmt: on


def is_manual_model(name: str) -> bool:
    return name in MANUAL_MODELS


def get_manual_model_info(name: str) -> MaseModelInfo:
    if name not in MANUAL_MODELS:
        raise ValueError(f"Manual model {name} is not supported")
    return MANUAL_MODELS[name]["info"]


def get_manual_model(
    name: str,
    task: str,
    dataset_info: dict,
    pretrained: bool,
    checkpoint: str | PathLike,
    quant_config: dict = None,
):
    """
    Args:
        name: The name of the model.
        task: The task type.
        dataset_info: The dataset info.
        pretrained: Whether to load the model dict.
        checkpoint: The checkpoint path (For HuggingFace Models this means both config and model dict).
        quant_config: The quantization config.

    ---
    Arg `pretrained` and `checkpoint`:
    - if pretrained and checkpoint: load pretrained config and model dict
    - if (not pretrained) and checkpoint: load pretrained config only, e.g., num_hidden_layers, num_attention_heads, etc.
    - else: raise RuntimeError
    """
    if name not in MANUAL_MODELS:
        raise ValueError(f"Manual model {name} is not supported")
    model_info: MaseModelInfo = MANUAL_MODELS[name]["info"]
    if model_info.is_quantized and quant_config is None:
        logger.warning(
            f"Model {name} is quantized but no quantization config is provided. Make sure you know what you are doing."
        )

    if task in ["cls", "classification"]:
        assert (
            model_info.sequence_classification
        ), f"Task {task} is not supported for {name}"
        if quant_config is not None:
            config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                checkpoint,
                quant_config=quant_config,
                num_labels=dataset_info["num_classes"],
            )
        else:
            config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                checkpoint,
                num_labels=dataset_info["num_classes"],
            )
        model_cls = MANUAL_MODELS[name]["sequence_classification"]
    elif task in ["lm", "language_modeling"]:
        assert model_info.causal_LM, f"Task {task} is not supported for {name}"
        if quant_config is not None:
            config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                checkpoint, quant_config=quant_config
            )
        else:
            config = MANUAL_MODELS[name]["config_cls"].from_pretrained(checkpoint)
        model_cls = MANUAL_MODELS[name]["causal_LM"]
    elif task in ["tran", "translation"]:
        assert model_info.seq2seqLM, f"Task {task} is not supported for {name}"
        if quant_config is not None:
            config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                checkpoint,
                quant_config=quant_config,
            )
        else:
            config = MANUAL_MODELS[name]["config_cls"].from_pretrained(checkpoint)
        model_cls = MANUAL_MODELS[name]["seq2seqLM"]
    else:
        raise ValueError(f"Task {task} is not supported for {name}")

    if pretrained:
        model = model_cls.from_pretrained(checkpoint, config=config)
    else:
        model = model_cls(config)

    return model


def get_manual_model_tokenizer(name: str, checkpoint: str | PathLike = None):
    if name not in MANUAL_MODELS:
        raise ValueError(f"Manual model {name} is not supported")
    return MANUAL_MODELS[name]["tokenizer_cls"].from_pretrained(
        name if checkpoint is None else checkpoint
    )


def get_manual_model_cls(name: str, task: str) -> type:
    if name not in MANUAL_MODELS:
        raise ValueError(f"Manual model {name} is not supported")
    if task in ["cls", "classification"]:
        hf_task = "sequence_classification"
    elif task in ["lm", "language_modeling"]:
        hf_task = "causal_LM"
    elif task in ["tran", "translation"]:
        hf_task = "seq2seqLM"
    else:
        raise ValueError(f"Task {task} is not supported")

    assert hf_task in MANUAL_MODELS[name], f"Task {task} is not supported for {name}"
    return MANUAL_MODELS[name][hf_task]


def get_manual_model_config_cls(name: str) -> type:
    if name not in MANUAL_MODELS:
        raise ValueError(f"Manual model {name} is not supported")
    return MANUAL_MODELS[name]["config_cls"]


def get_manual_model_tokenizer_cls(name: str) -> type:
    if name not in MANUAL_MODELS:
        raise ValueError(f"Manual model {name} is not supported")
    return MANUAL_MODELS[name]["tokenizer_cls"]
