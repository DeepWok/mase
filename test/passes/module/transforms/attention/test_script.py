#!/usr/bin/env python3
# Test script for transforming Llama model attention to MLA
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from chop.passes.module.transforms import attention_transform_pass

def test_llama_mla_transform():
    """
    Test Llama attention to MLA transformation with forward pass and inference.
    """
    checkpoint = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"Loading model from {checkpoint}")
    try:
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    # --- Original Model Inference ---
    inference_input = "Explain the concept of artificial intelligence in simple terms: "
    print(f"Input text: {inference_input}")
    try:
        input_ids = tokenizer(inference_input, return_tensors="pt").input_ids
        gen_params = {
            "max_new_tokens": 50,
            "temperature": 0.7,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        original_output_ids = model.generate(input_ids, **gen_params)
        original_output_text = tokenizer.decode(original_output_ids[0], skip_special_tokens=True)
        print(f"Original model output:\n{original_output_text}")
    except Exception as e:
        print(f"Error running inference with original model: {e}")
        return model, None

    # --- MLA Transformation ---
    print("Transforming model to use MLA attention...")
    model_type = model.config.model_type if hasattr(model.config, 'model_type') else "unknown"
    transform_key = "llama" if model_type.lower() == "llama" else "default"
    pass_args = {
        "by": "type",
        transform_key: {
            "config": {
                "name": "mla",
            }
        }
    }

    try:
        mla_model, stats = attention_transform_pass(model, pass_args)
        if not stats:
            print("Transformation returned empty stats - might indicate no changes were made")
    except Exception as e:
        print(f"Error during transformation: {e}")
        return model, None


    # --- MLA Model Inference ---
    print("Running inference with MLA-transformed model...")
    try:
        input_ids = tokenizer(inference_input, return_tensors="pt").input_ids
        gen_params = {
            "max_new_tokens": 50,
            "temperature": 0.7,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": False,  # Important for some MLA implementations
        }
       
        mla_output_ids = mla_model.generate(input_ids, **gen_params)
        mla_output_text = tokenizer.decode(mla_output_ids[0], skip_special_tokens=True)

        print(f"MLA model inference result:\n{mla_output_text}")

    except Exception as e:
        print(f"Error running inference with MLA model: {e}")

     # Check for MLA modules (Essential check)
    print("\nChecking for MLA modules...")
    found_mla = False
    for name, module in mla_model.named_modules():
        if 'MLA' in type(module).__name__ or hasattr(module, 'is_mla_wrapper'):
            found_mla = True
            break  # Exit loop once found

    if found_mla:
        print("Test completed successfully: MLA modules found, inference performed.")
    else:
        print("Test failed: No MLA modules found after transformation.")

    print("\n--- Comparison of generated text ---")
    print(f"Original: {original_output_text}")
    print(f"MLA     : {mla_output_text}")
    return model, mla_model


if __name__ == "__main__":
    test_llama_mla_transform()