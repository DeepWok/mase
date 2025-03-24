# Modified conversion script - handling shared tensors properly
import os
import json
import torch
import shutil
from pathlib import Path

# Path settings
input_dir = "/rds/general/user/yc3521/home/.llama/checkpoints/Llama3.2-3B"
output_dir = "/rds/general/user/yc3521/home/converted_llama32"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

print("="*50)
print(f"Converting model from {input_dir} to {output_dir}")
print("="*50)

# Read model params
try:
    with open(os.path.join(input_dir, "params.json"), "r") as f:
        params = json.load(f)
    print(f"Loaded model parameters: {params}")
except Exception as e:
    print(f"Error loading params.json: {e}")
    # Define sensible defaults for Llama 3.2 3B
    params = {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 32,
        "norm_eps": 1e-5,
        "vocab_size": 128256
    }
    print(f"Using default parameters: {params}")

# Create config.json for Hugging Face format
config = {
    "architectures": ["LlamaForCausalLM"],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": params.get("dim", 4096),
    "intermediate_size": 4 * params.get("dim", 4096),
    "num_attention_heads": params.get("n_heads", 32),
    "num_hidden_layers": params.get("n_layers", 32),
    "num_key_value_heads": params.get("n_kv_heads", params.get("n_heads", 32)),
    "rms_norm_eps": params.get("norm_eps", 1e-5),
    "rope_theta": params.get("rope_theta", 10000),
    "torch_dtype": "float16",
    "transformers_version": "4.33.0",
    "use_cache": True,
    "vocab_size": params.get("vocab_size", 128256),
    "model_type": "llama"
}

print("Writing config.json...")
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# Create minimal tokenizer files if original can't be used
print("Creating minimal tokenizer files...")

# Create a basic tokenizer_config.json
tokenizer_config = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "model_max_length": 2048,
    "tokenizer_class": "LlamaTokenizer",
    "unk_token": "<unk>"
}
with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
    json.dump(tokenizer_config, f, indent=2)

# Try to copy the tokenizer.model file, but don't worry if it fails
try:
    shutil.copy(os.path.join(input_dir, "tokenizer.model"), os.path.join(output_dir, "tokenizer.model"))
    print("Copied tokenizer.model file")
except Exception as e:
    print(f"Warning: Could not copy tokenizer.model: {e}")
    print("Will proceed with model conversion only")

# Load and convert the consolidated weights
print("\nLoading consolidated weights (this may take a while)...")
try:
    # Try to load with map_location to CPU to save memory
    consolidated_weights = torch.load(
        os.path.join(input_dir, "consolidated.00.pth"), 
        map_location="cpu"
    )
    print(f"Loaded weights with {len(consolidated_weights)} keys")
    print(f"First few keys: {list(consolidated_weights.keys())[:5]}")
except Exception as e:
    print(f"Error loading consolidated weights: {e}")
    print("Cannot proceed without model weights.")
    exit(1)

# Create a dictionary for the converted weights
pytorch_model = {}

print("\nConverting weights to Hugging Face format...")
# Map consolidated keys to HF keys
for key, value in consolidated_weights.items():
    hf_key = None
    # Process embeddings
    if key == "tok_embeddings.weight":
        hf_key = "model.embed_tokens.weight"
    # Process norm
    elif key == "norm.weight":
        hf_key = "model.norm.weight"
    # Process output - skip this for now to avoid shared tensor issue
    elif key == "output.weight":
        # We'll handle this separately
        continue
    # Process layers
    elif "layers" in key:
        parts = key.split(".")
        layer_idx = parts[1]
        layer_type = parts[2]
        
        if layer_type == "attention":
            # Handle attention submodules
            if parts[3] == "wq":
                hf_key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
            elif parts[3] == "wk":
                hf_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
            elif parts[3] == "wv":
                hf_key = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
            elif parts[3] == "wo":
                hf_key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        
        elif layer_type == "feed_forward":
            # Handle feed-forward submodules
            if parts[3] == "w1":
                hf_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
            elif parts[3] == "w2":
                hf_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
            elif parts[3] == "w3":
                hf_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
        
        elif layer_type == "attention_norm":
            hf_key = f"model.layers.{layer_idx}.input_layernorm.weight"
        
        elif layer_type == "ffn_norm":
            hf_key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
    
    # Add to PyTorch model if mapping exists
    if hf_key is not None:
        pytorch_model[hf_key] = value
    else:
        print(f"Warning: No mapping for key {key}")

# Now handle the output weight (lm_head) - make a COPY to avoid shared tensors
if "output.weight" in consolidated_weights:
    # Create a clone of the tensor to avoid shared memory
    pytorch_model["lm_head.weight"] = consolidated_weights["output.weight"].clone()

print(f"Converted {len(pytorch_model)} weight tensors")

# Save the model in the appropriate format
print("\nSaving model...")
try:
    # Create an index file for better loading
    index = {"weight_map": {}}
    for key in pytorch_model:
        # Save each tensor in a separate file
        filename = f"pytorch_model-{len(index['weight_map'])}.bin"
        index["weight_map"][key] = filename
        
        # Create a mini-dict with just this tensor
        mini_dict = {key: pytorch_model[key]}
        torch.save(mini_dict, os.path.join(output_dir, filename))
    
    # Save the index file
    with open(os.path.join(output_dir, "pytorch_model.bin.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    
    print(f"\nConversion complete! Model saved to {output_dir}")
    print("\nTo use this model:")
    print(f"model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
    print("Remember to use our SimpleTokenizer from earlier if the tokenizer doesn't work.")
    
except Exception as e:
    print(f"Error saving model: {e}")
    
    # Fallback to saving all weights in one file
    print("\nTrying fallback approach...")
    try:
        # Create a copy of the dictionary to avoid shared tensors
        safe_dict = {}
        for key, value in pytorch_model.items():
            safe_dict[key] = value.clone()
        
        torch.save(safe_dict, os.path.join(output_dir, "pytorch_model.bin"))
        print("Model saved using fallback approach")
    except Exception as e2:
        print(f"Fallback approach also failed: {e2}")