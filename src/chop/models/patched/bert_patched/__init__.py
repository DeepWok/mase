from transformers.models.bert import (
    BertConfig,
    BertForSequenceClassification,
    BertLMHeadModel,
    BertModel,
)
from transformers.models.bert.modeling_bert import BertLMHeadModel

from .configuration_bert_patched import BertConfigPatched
from .modeling_bert_patched import BertLMHeadModelPatched, BertModelPatched

bert_patched_model_cls_to_required_input_args = {
    BertModelPatched: ["input_ids", "attention_mask", "token_type_ids"],
    # BertLMHeadModelPatched: ["input_ids", "attention_mask", "token_type_ids", "labels"],
}

# load pretrained from original model/config
bert_patched_cls_to_original_cls = {
    BertConfigPatched: BertConfig,
    BertModelPatched: BertModel,
    # BertLMHeadModelPatched: BertLMHeadModel,
}

# used in nlp cls model wrapper, get output from base model and feed it to classifier
bert_patched_name_to_pooler_output_name = {
    "bert-base-uncased@patched": "pooler_output",
    "bert-large-uncased@patched": "pooler_output",
    "bert-base-cased@patched": "pooler_output",
    "bert-large-cased@patched": "pooler_output",
}

# used in get_model_mapping in patched_nlp_models.py
_bert_patched_task_to_model_cls = {
    "tokenizer": None,
    "config": BertConfigPatched,
    "base": BertModelPatched,
    # "lm": BertLMHeadModelPatched,
    "cls": BertModelPatched,
}

bert_patched_name_to_patched_model_mapping = {
    "bert-base-uncased@patched": _bert_patched_task_to_model_cls,
    "bert-large-uncased@patched": _bert_patched_task_to_model_cls,
    "bert-base-cased@patched": _bert_patched_task_to_model_cls,
    "bert-large-cased@patched": _bert_patched_task_to_model_cls,
}

# create classifier (Linear) in get_patched_nlp_model
bert_patched_model_name_to_hidden_size = {
    "bert-base-uncased@patched": 768,
    "bert-base-cased@patched": 768,
    "bert-large-uncased@patched": 1024,
    "bert-large-cased@patched": 1024,
}

# used in get_patched_nlp_model to copy pretrained weight into patched one
_task_to_original_cls = {
    "cls": BertModel
    # "lm": BertLMHeadModel,
}
bert_patched_model_name_to_task_to_original_cls = {
    "bert-base-uncased@patched": _task_to_original_cls,
    "bert-large-uncased@patched": _task_to_original_cls,
    "bert-base-cased@patched": _task_to_original_cls,
    "bert-large-cased@patched": _task_to_original_cls,
}
