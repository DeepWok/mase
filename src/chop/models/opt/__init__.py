from transformers.models.gpt2 import GPT2Tokenizer
from .configuration_opt import OPTPatchedConfig
from .modeling_opt import (
    OPTPatchedForCausalLM,
    OPTPatchedModel,
    OPTPatchedForSequenceClassification,
)

# required input args will not be provided in dummy inputs before mase_symbolic_trace
opt_patched_model_cls_to_required_input_args = {
    OPTPatchedModel: ["input_ids", "attention_mask"],
    OPTPatchedForCausalLM: ["input_ids", "attention_mask", "labels"],
}
