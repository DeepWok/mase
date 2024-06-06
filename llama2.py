import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import evaluate


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using", device)

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

    # def tokenize_function(example):
    #     return tokenizer(example["sentence"], truncation=True)

    # tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt")
    print(model_inputs)

    generated_ids = model.generate(**model_inputs)
    print(generated_ids)

    out = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(out)


if __name__ == "__main__":
    glue_sst2 = load_dataset("nyu-mll/glue", "sst2")
    pretrained_llama2(glue_sst2)
