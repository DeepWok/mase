from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorWithPadding,
)
from datasets import load_dataset


def training_llama2():
    """
    Training on Llama-2-7B on a dataset to see effects of varying num KV heads
    on the accuracy of the model.
    """
    cfg = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_config(cfg)


def pretrained_llama2(dataset):
    """
    Taking pretrained Llama-2-7B and using mean weights as described by GQA paper.
    """
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

    def tokenize_function(example):
        return tokenizer(example["sentence"], truncation=True, max_length=model)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


if __name__ == "__main__":
    glue_sst2 = load_dataset("nyu-mll/glue", "sst2")
    pretrained_llama2(glue_sst2)
