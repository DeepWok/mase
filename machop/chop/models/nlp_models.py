import os

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

model_name_to_hidden_size = {
    "facebook/opt-125m": 768,
    "facebook/opt-350m": 1024,
    "facebook/opt-1.3b": 2048,
    "facebook/opt-2.7b": 2560,
    "facebook/opt-6.7b": 4096,
    "facebook/opt-13b": 5120,
    "facebook/opt-30b": 7168,
    "facebook/opt-66b": 9126,
}

model_name_to_pooler_size = {
    "facebook/opt-125m": (768, 768),
    "facebook/opt-350m": (512, 1024),
    "facebook/opt-1.3b": (2048, 2048),
    "facebook/opt-2.7b": (2560, 2560),
    "facebook/opt-6.7b": (4096, 4096),
    "facebook/opt-13b": (5120, 5120),
    "facebook/opt-30b": (7168, 7168),
    "facebook/opt-66b": (9126, 9126),
}


# TODO: check the pooler? should we pool at the first token for opt?
# bert uses the pooler
# opt does not use the pooler
class Pooler(nn.Module):
    def __init__(self, in_hidden_size, out_hidden_size):
        super().__init__()
        self.dense = nn.Linear(in_hidden_size, out_hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def get_nlp_model(name, task, info, checkpoint=None, pretrained=True):
    if task not in [
        "classification",
        "cls",
        "translation",
        "tran",
        "language_modeling",
        "lm",
    ]:
        raise ValueError("task must be a valid value for NLP models")

    if task in ["classification", "cls"]:
        num_classes = info["num_classes"]
    tokenizer = AutoTokenizer.from_pretrained(
        name, cache_dir=os.path.abspath("./cache/tokenizer_cache_dir"), return_dict=True
    )
    if pretrained:
        print(f"Loaded tokenizer from {name}")
        if checkpoint is not None:
            # Load checkpoint if `--pretrained --load LOAD`
            model = AutoModel.from_pretrained(checkpoint)
            print(f"Loaded local pretrained HuggingFace model from {checkpoint}")
        else:
            if task in ["language_modeling", "lm"]:
                model = AutoModelForCausalLM.from_pretrained(
                    name,
                    return_dict=True,
                    cache_dir=os.path.abspath("./cache/model_cache_dir"),
                )
            elif task in ["translation", "tran"]:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    name,
                    return_dict=True,
                    cache_dir=os.path.abspath("./cache/model_cache_dir"),
                )
            else:
                # cls task
                model = AutoModel.from_pretrained(
                    name,
                    return_dict=True,
                    cache_dir=os.path.abspath("./cache/model_cache_dir"),
                )
            print(f"Loaded pretrained model using model name '{name}' in HuggingFace")
    else:
        config = AutoConfig.from_pretrained(name)
        if task in ["classification", "cls"]:
            model = AutoModel.from_config(config=config)
        elif task in ["language_modeling", "lm"]:
            raise ValueError(
                "Language modeling task is not supported to train from scratch, please use --pretrained flag"
            )
        elif task == ["translation", "tran"]:
            model = AutoModelForSeq2SeqLM.from_config(config=config)
        print("HuggingFace model randomly initialized")

    if task in ["classification", "cls"]:
        hidden_size = model_name_to_hidden_size.get(name, model.config.hidden_size)
        classifier = nn.Linear(hidden_size, num_classes)
        if name in model_name_to_pooler_size:
            in_hidden, out_hidden = model_name_to_pooler_size[name]
            pooler = Pooler(in_hidden, out_hidden)
            classifier = nn.Sequential(pooler, classifier)
    else:
        classifier = None
    return {
        "model": model,
        "tokenizer": tokenizer,
        "classifier": classifier,
    }
