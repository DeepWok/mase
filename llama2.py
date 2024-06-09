from pathlib import Path
import math
from datetime import datetime
import os
import random

import torch
from torch import nn

import numpy as np
import pandas as pd

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    LlamaModel,
    LlamaForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaSdpaAttention
from datasets import load_dataset
import evaluate

checkpoint = "JackFram/llama-160m"
print("Num GPUs:", torch.cuda.device_count())
device = torch.device(0)

def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def training_small_llama2():
    """
    Training on small Llama-2 style model on a dataset to see effects of
    varying num KV heads on the accuracy of the model.
    """
    cfg = AutoConfig.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
    )
    # Lower the number of layers
    cfg.num_hidden_layers = 4

    # Adjust number of kv heads
    # cfg.num_key_value_heads = 1, 2, 4, 8, 16, 32 ...

    print("Llama-2 Config")
    print(cfg)

    model = AutoModelForCausalLM.from_config(cfg)
    print("Llama-2 Model")
    print(model)

    print("Number of params:", num_params(model))


def group_texts(
    examples,
    block_size: int = 128,
):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def transform_weights_gqa(
    model: LlamaForCausalLM,
    new_num_kv_heads: int,
):
    """
    Transform a multi-head attention Llama into a GQA one with new_num_kv_heads.
    """

    # Extract info from current model
    self_attn_module = model.model.layers[0].self_attn
    hidden_size = self_attn_module.hidden_size
    num_heads = self_attn_module.num_heads
    head_dim = self_attn_module.head_dim

    def _transform_llama_linear_weights(weights):
        # (num_kv_heads, group_size, head_dim, hidden_size)
        grouped_weights = weights.reshape(new_num_kv_heads, -1, head_dim, hidden_size)
        # Average across group_size dim
        # (num_kv_heads, head_dim, hidden_size)
        gqa_weights = grouped_weights.mean(dim=1)
        # Change to 2D
        # (num_kv_heads * head_dim, hidden_size)
        gqa_weights = gqa_weights.reshape(new_num_kv_heads * head_dim, hidden_size)
        return gqa_weights


    with torch.no_grad():
        for llama_decoder_layer in model.model.layers:

            old_self_attn = llama_decoder_layer.self_attn

            new_cfg = old_self_attn.config
            new_cfg.num_key_value_heads = new_num_kv_heads

            # Make new attention layer
            new_self_attn = LlamaSdpaAttention(
                config=new_cfg,
                layer_idx=old_self_attn.layer_idx,
            )

            # Same Q & O projections
            new_self_attn.q_proj.weight = old_self_attn.q_proj.weight
            new_self_attn.o_proj.weight = old_self_attn.o_proj.weight

            # Replace K Projection into GQA
            k_weights = old_self_attn.k_proj.weight
            new_k_weights = _transform_llama_linear_weights(k_weights)
            new_self_attn.k_proj.weight[:,:] = new_k_weights

            # Make V Projection into GQA
            v_weights = old_self_attn.v_proj.weight
            new_v_weights = _transform_llama_linear_weights(v_weights)
            new_self_attn.v_proj.weight[:,:] = new_v_weights

            # Assign back
            llama_decoder_layer.self_attn = new_self_attn

    model = model.cuda()
    return model


def eli5_dataset():
    """5000 Rows of ELI5 Category Dataset"""

    # Tokenizer & Data Collator
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Tokenize Dataset
    raw_dataset = load_dataset(
        "eli5_category",
        split="train[:5000]",
        trust_remote_code=True,
    )
    raw_dataset = raw_dataset.train_test_split(test_size=0.2)
    raw_dataset = raw_dataset.flatten()

    def _preprocess_fn(examples):
        return tokenizer([" ".join(x) for x in examples["answers.text"]])

    tokenized_inputs = raw_dataset.map(
        _preprocess_fn,
        batched=True,
        num_proc=4,
        # Remove all original cols, leaving only new tokenizer cols
        remove_columns=raw_dataset["train"].column_names,
    )

    model_inputs = tokenized_inputs.map(
        group_texts,
        batched=True,
        num_proc=4,
        fn_kwargs=dict(
            block_size=128,
        ),
    )

    return model_inputs, data_collator


def billsum_dataset():
    billsum = load_dataset("billsum", split="ca_test")
    billsum = billsum.train_test_split(test_size=0.2)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    prefix = "Summarize: "

    def _preprocess_fn(examples):
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    model_inputs = billsum.map(
        _preprocess_fn,
        batched=True,
        num_proc=4,
        remove_columns=billsum["train"].features,
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

    return model_inputs, data_collator

def compute_metrics(tokenizer, eval_pred):
    rouge = evaluate.load("rouge")
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def sweep_pretrained_llama2_language_modelling(
    output_dir: Path,
    task: str,
    choices_kv_heads: list[int] = [1, 2, 3, 4, 6, 12],
    learning_rate = 2e-5,
    weight_decay = 0.01,
    seed = random.randint(0, 2**30),
):
    """
    Taking pretrained Llama-2-7B and using mean weights as described by GQA paper.
    """

    _tasks = ["language_modelling", "summarization"]
    assert task in _tasks, f"<task> arg needs to be one of {_tasks}!"

    if task == "language_modelling":
        model_inputs, data_collator = eli5_dataset()
    elif task == "summarization":
        model_inputs, data_collator = billsum_dataset()

    sweep_data = []

    # for kv_heads in choices_kv_heads:
    for kv_heads in choices_kv_heads:

        print(f"#### TEST: KV_HEADS={kv_heads}, seed: {seed}")

        # Load model with desired number of kv heads
        model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            checkpoint,
        )
        model = transform_weights_gqa(model, new_num_kv_heads=kv_heads)
        model.to(device=device)

        # Setup data gathering
        self_attn = model.model.layers[0].self_attn
        data_record = {
            "task": task,
            "seed": seed,
            "vocab_size": model.model.vocab_size,
            "attention_dropout": self_attn.attention_dropout,
            "hidden_size": self_attn.hidden_size,
            "num_heads": self_attn.num_heads,
            "head_dim": self_attn.head_dim,
            "num_key_value_heads": self_attn.num_key_value_heads,
            "num_key_value_groups": self_attn.num_key_value_groups,
            "max_position_embeddings": self_attn.max_position_embeddings,
            "rope_theta": self_attn.rope_theta,
            "is_causal": self_attn.is_causal,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }

        # Setup Trainer
        working_dir = output_dir / f"{kv_heads}_kv_heads"
        os.makedirs(working_dir, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=str(working_dir),
            eval_strategy="epoch",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            per_device_train_batch_size=8,
            push_to_hub=False,
            report_to="none",
            seed=seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=model_inputs["train"],
            eval_dataset=model_inputs["test"],
            data_collator=data_collator,
        )

        # Uptrain model if not MHA
        if kv_heads != 12:
            trainer.train()

        eval_results = trainer.evaluate()

        data_record.update({
            "eval_loss": eval_results['eval_loss'],
            "perplexity": math.exp(eval_results['eval_loss']),
        })

        print(data_record)
        sweep_data.append(data_record)

    return pd.DataFrame.from_records(sweep_data)


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "output"
    timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    timestamp_dir = output_dir / timestamp
    os.makedirs(timestamp_dir, exist_ok=True)

    # Run
    data = sweep_pretrained_llama2_language_modelling(
        output_dir=timestamp_dir,
        task="language_modelling",
    )
    data.to_csv(timestamp_dir / f"llama2_kv_head_sweep_{timestamp}.csv")
