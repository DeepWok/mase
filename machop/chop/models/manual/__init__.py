from os import PathLike
import logging
from ..utils import MaseModelInfo
from .bert_quantized import (
    BertQuantizedConfig,
    BertQuantizedForSequenceClassification,
    BertTokenizer,
    parse_bert_quantized_config,
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
    parse_llama_quantized_config,
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
    parse_opt_quantized_config,
)

logger = logging.getLogger(__name__)

# fmt: off
MANUAL_MODELS = {
    "bert_quantized": {
        "config_cls": BertQuantizedConfig,
        "tokenizer_cls": BertTokenizer,
        "info": MaseModelInfo("bert_quantized", model_source="manual", task_type="nlp", sequence_classification=True, is_quantized=True),
        "sequence_classification": BertQuantizedForSequenceClassification,
        "quant_config_parser": parse_bert_quantized_config,
    },
    "llama_plain": {
        "config_cls": LlamaConfig,
        "tokenizer_cls": LlamaTokenizer,
        "info": MaseModelInfo("llama_plain", model_source="manual", task_type="nlp", sequence_classification=True, causal_LM=True),
        "sequence_classification": LlamaForSequenceClassification,
        "causal_LM": LlamaForCausalLM,
    },
    "llama_quantized": {
        "config_cls": LlamaQuantizedConfig,
        "tokenizer_cls": LlamaTokenizer,
        "info": MaseModelInfo("llama_quantized", model_source="manual", task_type="nlp", sequence_classification=True, causal_LM=True, is_quantized=True),
        "sequence_classification": LlamaQuantizedForSequenceClassification,
        "causal_LM": LlamaQuantizedForCausalLM,
        "quant_config_parser": parse_llama_quantized_config,
    },
    "opt_plain": {
        "config_cls": OPTConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": MaseModelInfo("opt_plain", model_source="manual", task_type="nlp", sequence_classification=True, causal_LM=True),
        "sequence_classification": OPTForSequenceClassification,
        "causal_LM": OPTForCausalLM,
    },
    "opt_quantized": {
        "config_cls": OPTQuantizedConfig,
        "tokenizer_cls": GPT2Tokenizer,
        "info": MaseModelInfo("opt_quantized", model_source="manual", task_type="nlp", sequence_classification=True, causal_LM=True, is_quantized=True),
        "sequence_classification": OPTQuantizedForSequenceClassification,
        "causal_LM": OPTQuantizedForCausalLM,
        "quant_config_parser": parse_opt_quantized_config,
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
        logger.info(
            f"Model {name} is quantized but no quantization config is provided."
        )

    if task in ["cls", "classification"]:
        assert (
            model_info.sequence_classification
        ), f"Task {task} is not supported for {name}"
        if quant_config is not None:
            config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                checkpoint,
                quant_config=quant_config,
                num_labels=dataset_info.num_classes,
            )
        else:
            config = MANUAL_MODELS[name]["config_cls"].from_pretrained(
                checkpoint,
                num_labels=dataset_info.num_classes,
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
        logger.info(f"Manual model's state_dict is loaded from {checkpoint}")
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


def get_manual_model_quant_config_parser(
    name: str = None, config_cls: type = None
) -> callable:
    if name is None and config_cls is None:
        raise ValueError("Must provide either name or config_cls")

    if name is not None:
        if name not in MANUAL_MODELS:
            raise ValueError(f"Manual model {name} is not supported")
        model_info = MANUAL_MODELS[name]["info"]
        if not model_info.is_quantized:
            raise ValueError(f"Model {name} is not quantized")

        return MANUAL_MODELS[name]["quant_config_parser"]
    elif config_cls is not None:
        for model_name, model_meta in MANUAL_MODELS.items():
            if model_meta["config_cls"] == config_cls:
                model_info = model_meta["info"]
                if not model_info.is_quantized:
                    raise ValueError(f"Model {model_name} is not quantized")
                return model_meta["quant_config_parser"]
        raise ValueError(f"Model {config_cls} is not supported")
    else:
        raise ValueError("Must provide either name or config_cls")
