#!/usr/bin/env python3
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import chop.passes as passes
from chop.passes.module.transforms.attention import fc_transform_pass


def debug_fc_replacement():
    # original gpt2
    original_model = GPT2LMHeadModel.from_pretrained("gpt2")
    original_model.eval()

    # FCed gpt2
    fc_model = GPT2LMHeadModel.from_pretrained("gpt2")
    fc_model.eval()
    module_name = "transformer.h.11.attn"
    fc_model = fc_transform_pass(fc_model, module_name, config={})

    # input
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    input_text = "Hello world"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    with torch.no_grad():
        original_outputs = original_model(input_ids, output_attentions=True)
        fc_outputs = fc_model(input_ids, output_attentions=True)

    # original_outputs.logits: (batch_size, seq_len, vocab_size)
    print("===> Last token logits (first 10 dims) <===")
    print("Original logits:", original_outputs.logits[0, -1, :10])
    print("FC replaced logits:", fc_outputs.logits[0, -1, :10])

    diff = (original_outputs.logits - fc_outputs.logits).abs().mean()
    print(f"Mean difference in logits: {diff.item():.6f}")

    print("\n===> Compare generation <===")
    max_length = 30
    with torch.no_grad():
        original_gen_ids = original_model.generate(
            input_ids, attention_mask=attention_mask, max_length=max_length
        )
        fc_gen_ids = fc_model.generate(
            input_ids, attention_mask=attention_mask, max_length=max_length
        )

    original_text = tokenizer.decode(original_gen_ids[0], skip_special_tokens=True)
    fc_text = tokenizer.decode(fc_gen_ids[0], skip_special_tokens=True)

    print("Original generation:", original_text)
    print("FC replaced generation:", fc_text)


if __name__ == "__main__":
    debug_fc_replacement()
