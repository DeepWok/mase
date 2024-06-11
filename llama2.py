from pathlib import Path
import math
from datetime import datetime
import os
import random
from typing import Optional, Tuple
import re

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
from transformers.models.llama.modeling_llama import (
    LlamaSdpaAttention,
    LlamaConfig,
    LlamaRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from datasets import load_dataset
import evaluate

from chop.nn.quantized.modules import GroupedQueryAttentionInteger


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


class LlamaHWGQA(GroupedQueryAttentionInteger):
    """Wrapper module to get GQA Integer working with hugging face Llama Models."""

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: Optional[int] = None,
        linear_q_config: dict = None,
        linear_out_q_config: dict = None,
        softermax_out_q_config: dict = None,
        qk_matmul_out_q_config: dict = None,
        v_matmul_out_q_config: dict = None,
    ) -> None:

        super().__init__(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            bias=False,
            linear_q_config=linear_q_config,
            linear_out_q_config=linear_out_q_config,
            softermax_out_q_config=softermax_out_q_config,
            qk_matmul_out_q_config=qk_matmul_out_q_config,
            v_matmul_out_q_config=v_matmul_out_q_config,
            floor=True,
        )

        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states, key_states, value_states = self._qkv_states(hidden_states, bsz, q_len)

        # query_states = self.q_proj(hidden_states)
        # key_states = self.k_proj(hidden_states)
        # value_states = self.v_proj(hidden_states)

        # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output = self._attention_mechanism(
            query_states, key_states, value_states, bsz, q_len
        )
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        # causal_mask = attention_mask
        # if attention_mask is not None:
        #     causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # # Reference: https://github.com/pytorch/pytorch/issues/112577.
        # if query_states.device.type == "cuda" and causal_mask is not None:
        #     query_states = query_states.contiguous()
        #     key_states = key_states.contiguous()
        #     value_states = value_states.contiguous()

        # # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
        # # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
        # is_causal = True if causal_mask is None and q_len > 1 else False

        # attn_output = torch.nn.functional.scaled_dot_product_attention(
        #     query_states,
        #     key_states,
        #     value_states,
        #     attn_mask=causal_mask,
        #     dropout_p=self.attention_dropout if self.training else 0.0,
        #     is_causal=is_causal,
        # )

        # attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        # attn_output = self.o_proj(attn_output)

        return attn_output, None, None



def transform_hardware_gqa(
    model: LlamaForCausalLM,
    linear_q_config: dict,
    linear_out_q_config: dict,
    softermax_out_q_config: dict,
    qk_matmul_out_q_config: dict,
    v_matmul_out_q_config: dict,
):
    """
    Transforms all Sdpa Attention modules into MASE GroupedQueryAttentionInteger
    """
    with torch.no_grad():
        for llama_decoder_layer in model.model.layers:

            old_attn = llama_decoder_layer.self_attn
            cfg = old_attn.config

            hw_gqa = LlamaHWGQA(
                config=cfg,
                layer_idx=old_attn.layer_idx,
                linear_q_config=linear_q_config,
                linear_out_q_config=linear_out_q_config,
                softermax_out_q_config=softermax_out_q_config,
                qk_matmul_out_q_config=qk_matmul_out_q_config,
                v_matmul_out_q_config=v_matmul_out_q_config,
            )

            llama_decoder_layer.self_attn = hw_gqa

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

        trainer.train()

        eval_results = trainer.evaluate()

        data_record.update({
            "eval_loss": eval_results['eval_loss'],
            "perplexity": math.exp(eval_results['eval_loss']),
        })

        print(data_record)
        sweep_data.append(data_record)

    return pd.DataFrame.from_records(sweep_data)


def uptrain_kv_heads_software():
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
    return data


def inference_accuracy_eval_llama2(
    sweep_dir: Path,
    width: int = 30,
    frac_width: int = 10,
):
    model_inputs, data_collator = eli5_dataset()

    eval_data = []

    for kv_head_dir in sweep_dir.glob("*_kv_heads"):
        kv_heads = int(re.search(r"(\d+)_kv_heads", str(kv_head_dir)).groups()[0])
        chkpt = kv_head_dir / "checkpoint-4000"

        model = AutoModelForCausalLM.from_pretrained(
            chkpt,
            use_cache=False,
        )

        linear_q_config = {
            "data_in_width": width,
            "data_in_frac_width": frac_width,
            "weight_width": width,
            "weight_frac_width": frac_width,
            "bias_width": width,
            "bias_frac_width": frac_width,
        }

        # All have same output width configuration
        linear_out_q_config = {
            "data_out_width": width,
            "data_out_frac_width": frac_width,
        }
        qk_matmul_out_q_config = linear_out_q_config
        v_matmul_out_q_config = linear_out_q_config
        softermax_out_q_config = {
            "width": linear_out_q_config["data_out_width"],
            "frac_width": linear_out_q_config["data_out_frac_width"],
        }

        # linear_q_config = {
        #     "data_in_width": 8,
        #     "data_in_frac_width": 4,
        #     "weight_width": 8,
        #     "weight_frac_width": 4,
        #     "bias_width": 8,
        #     "bias_frac_width": 4,
        # }

        # # All have same output width configuration
        # linear_out_q_config = {
        #     "data_out_width": 12,
        #     "data_out_frac_width": 4,
        # }
        # qk_matmul_out_q_config = {
        #     "data_out_width": 16,
        #     "data_out_frac_width": 4,
        # }
        # softermax_out_q_config = {
        #     "width": qk_matmul_out_q_config["data_out_width"],
        #     "frac_width": qk_matmul_out_q_config["data_out_frac_width"],
        # }
        # v_matmul_out_q_config = {
        #     "data_out_width": 20,
        #     "data_out_frac_width": 4,
        # }

        model = transform_hardware_gqa(
            model=model,
            linear_q_config=linear_q_config,
            linear_out_q_config=linear_out_q_config,
            softermax_out_q_config=softermax_out_q_config,
            qk_matmul_out_q_config=qk_matmul_out_q_config,
            v_matmul_out_q_config=v_matmul_out_q_config,
        )
        model.to(device=device)

        # Data gathering
        self_attn = model.model.layers[0].self_attn
        data_record = {
            "kv_heads": kv_heads,
            "width": width,
            "frac_width": frac_width,
            "hidden_size": self_attn.hidden_size,
            "num_heads": self_attn.num_heads,
            "head_dim": self_attn.head_dim,
        }

        # Evaluation
        training_args = TrainingArguments(
            output_dir=str(kv_head_dir),
            eval_strategy="epoch",
            push_to_hub=False,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=model_inputs["train"],
            eval_dataset=model_inputs["test"],
            data_collator=data_collator,
        )

        eval_results = trainer.evaluate()

        data_record.update({
            "eval_loss": eval_results['eval_loss'],
            "perplexity": math.exp(eval_results['eval_loss']),
        })

        print(data_record)

        eval_data.append(data_record)

    return pd.DataFrame.from_records(eval_data).sort_values(by="kv_heads")


if __name__ == "__main__":
    # Uptraining
    # train_data = uptrain_kv_heads_software()
    # print(train_data)

    # HW Inference
    sweep_dir = Path(__file__).parent / "output/archive/full_head_sweep"
    eval_data = inference_accuracy_eval_llama2(sweep_dir)
    print(eval_data)
