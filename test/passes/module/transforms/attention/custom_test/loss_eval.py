import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from llama.modeling_llama import LlamaForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import copy
import gc
import math
import time

def print_separator():
    print("=" * 80)

def apply_transformations(model):
    """Apply the three stages of transformations to the model."""
    # Make a deep copy to modify
    model_copy = copy.deepcopy(model)
    
    hidden_size = model_copy.config.hidden_size
    n_heads = model_copy.config.num_attention_heads
    kv_heads = model_copy.config.num_key_value_heads
    head_dim = model_copy.config.hidden_size // model_copy.config.num_attention_heads
    latent_dim = kv_heads * head_dim
    kv_groups = model_copy.config.num_attention_heads // kv_heads if kv_heads > 0 else 1
    
    print(f"\nModel architecture details:")
    print(f"hidden_size: {hidden_size}, n_heads: {n_heads}, kv_heads: {kv_heads}")
    print(f"head_dim: {head_dim}, latent_dim: {latent_dim}, kv_groups: {kv_groups}")
    
    # Stage 1: Initial Weight Modification (Identity Matrices)
    print("\n--- Applying Initial Weight Modification (Identity) ---")
    modified_layers = 0
    
    with torch.no_grad():
        for name, module in model_copy.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                target_device = module.weight.data.device
                target_dtype = module.weight.data.dtype
                if 'k_up_proj' in name or "v_up_proj" in name:
                    modified_layers += 1
                    if modified_layers <= 2:
                        print(f"Modifying layer: {name}")
                    
                    # Constructing identity matrix based on dimensions
                    identity_weight = torch.stack(
                        [torch.eye(latent_dim).reshape(kv_heads, head_dim, latent_dim)] * kv_groups,
                        dim=1
                    ).reshape(hidden_size, latent_dim).contiguous().to(target_device, target_dtype)
                    
                    if 'k_up_proj' in name:
                        # Reshape/transpose specific to k_up_proj
                        identity_weight = identity_weight.view(hidden_size, kv_heads, head_dim).transpose(1, 2).contiguous().view(hidden_size, latent_dim)
                    
                    module.weight.data.copy_(identity_weight)
                
                elif 'k_proj' in name:  # Apply reshaping to k_proj weights and bias
                    # Reshape weight
                    reshaped_weight = module.weight.data.view(kv_heads, head_dim, hidden_size).transpose(0, 1).contiguous().view(latent_dim, hidden_size)
                    module.weight.data.copy_(reshaped_weight)
                    
                    # Reshape bias if it exists
                    if hasattr(module, 'bias') and module.bias is not None:
                        reshaped_bias = module.bias.data.view(kv_heads, head_dim).transpose(0, 1).contiguous().view(latent_dim)
                        module.bias.data.copy_(reshaped_bias)
    
    print(f"Initial modification complete. Modified {modified_layers} layers.")
    
    # Stage 2: Second Weight Modification (Orthogonalization via SVD)
    print("\n--- Applying Second Weight Modification (Orthogonalization) ---")
    modified_layers = 0
    
    with torch.no_grad():
        for name, module in model_copy.named_modules():
            # Check if the module is a self-attention layer
            if isinstance(module, torch.nn.Module) and "self_attn" in name and hasattr(module, 'k_up_proj'):
                modified_layers += 1
                if modified_layers <= 2:
                    print(f"Orthogonalizing layer: {name}")
                
                target_device = module.q_proj.weight.device
                target_dtype = module.q_proj.weight.dtype
                
                # Orthogonalize q_proj and k_up_proj
                k_up_weight = module.k_up_proj.weight.data.clone().reshape(n_heads, head_dim, latent_dim)
                q_weight = module.q_proj.weight.data.clone().reshape(n_heads, head_dim, hidden_size)
                
                if module.q_proj.bias is not None:
                    q_bias = module.q_proj.bias.data.clone().reshape(n_heads, head_dim, 1)
                    q_weight = torch.cat([q_weight, q_bias], dim=-1)  # Append bias as a column
                
                q_k_up = torch.einsum("hdc,hdD->hcD", k_up_weight, q_weight)
                
                # SVD - Use torch.linalg.svd for stability
                U, S, Vh = torch.linalg.svd(q_k_up, full_matrices=False)
                V = Vh.mH  # Conjugate transpose for V
                
                # Keep only top 'head_dim' components
                U = U[:, :, :head_dim]
                S = S[:, :head_dim]
                V = V[:, :, :head_dim]
                
                S_sqrt = torch.sqrt(S)
                US_sqrt = torch.einsum('hLd, hd->hdL', U, S_sqrt)
                S_sqrtV = torch.einsum('hd, hdD->hdD', S_sqrt, V.mH)
                
                # Update weights and bias
                module.k_up_proj.weight.data.copy_(US_sqrt.reshape(n_heads * head_dim, latent_dim).contiguous())
                
                if module.q_proj.bias is not None:
                    module.q_proj.bias.data.copy_(S_sqrtV[:, :, -1].reshape(-1).contiguous())
                    S_sqrtV_weights = S_sqrtV[:, :, :-1]  # Separate weights from bias column
                else:
                    S_sqrtV_weights = S_sqrtV
                
                module.q_proj.weight.data.copy_(S_sqrtV_weights.reshape(n_heads * head_dim, hidden_size).contiguous())
                
                # Orthogonalize o_proj and v_up_proj
                v_up_weight = module.v_up_proj.weight.data.clone().reshape(n_heads, head_dim, latent_dim)
                o_weight = module.o_proj.weight.data.clone().reshape(hidden_size, n_heads, head_dim)
                v_up_o = torch.einsum("hdc,Dhd->hcD", v_up_weight, o_weight)
                
                # SVD
                U, S, Vh = torch.linalg.svd(v_up_o, full_matrices=False)
                V = Vh.mH
                
                # Keep only top 'head_dim' components
                U = U[:, :, :head_dim]
                S = S[:, :head_dim]
                V = V[:, :, :head_dim]
                
                S_sqrt = torch.sqrt(S)
                US_sqrt = torch.einsum('hLd, hd->hdL', U, S_sqrt)
                S_sqrtV = torch.einsum('hd, hDd->Dhd', S_sqrt, V)
                
                # Update weights
                module.v_up_proj.weight.data.copy_(US_sqrt.reshape(n_heads*head_dim, latent_dim).contiguous())
                module.o_proj.weight.data.copy_(S_sqrtV.reshape(hidden_size, n_heads * head_dim).contiguous())
    
    print(f"Orthogonalization complete. Modified {modified_layers} layers.")
    
    # Stage 3: Third Weight Modification (Absorption)
    print("\n--- Applying Third Weight Modification (Absorption) ---")
    
    with torch.no_grad():
        layers_to_modify = []
        # First, identify layers that still need absorption
        for name, module in model_copy.named_modules():
            if "self_attn" in name and hasattr(module, 'k_up_proj') and hasattr(module, 'v_up_proj'):
                layers_to_modify.append((name, module))
        
        print(f"Found {len(layers_to_modify)} layers to absorb")
        
        # Now, modify them. This avoids issues with modifying modules while iterating.
        for idx, (name, module) in enumerate(layers_to_modify):
            if idx <= 1:
                print(f"Absorbing layer: {name}")
                
            target_device = module.q_proj.weight.device
            target_dtype = module.q_proj.weight.dtype
            
            # Absorb k_up_proj into q_proj
            k_up_weight = module.k_up_proj.weight.data.clone().reshape(n_heads, head_dim, latent_dim)
            q_weight = module.q_proj.weight.data.clone().reshape(n_heads, head_dim, hidden_size)
            q_bias_data = None
            
            if module.q_proj.bias is not None:
                q_bias = module.q_proj.bias.data.clone().reshape(n_heads, head_dim, 1)
                q_weight = torch.cat([q_weight, q_bias], dim=-1)  # Append bias column
            
            q_k_up = torch.einsum("hdc,hdD->hcD", k_up_weight, q_weight)
            
            # Create new linear layer for absorbed q_proj
            new_q_proj_out_features = n_heads * latent_dim
            new_q_proj = torch.nn.Linear(hidden_size, new_q_proj_out_features, bias=(module.q_proj.bias is not None))
            new_q_proj = new_q_proj.to(device=target_device, dtype=target_dtype)
            
            if module.q_proj.bias is not None:
                new_q_proj.bias.data.copy_(q_k_up[:, :, -1].reshape(-1).contiguous())
                q_k_up_weights = q_k_up[:, :, :-1]  # Separate weights
            else:
                q_k_up_weights = q_k_up
            
            new_q_proj.weight.data.copy_(q_k_up_weights.reshape(new_q_proj_out_features, hidden_size).contiguous())
            
            # Replace module's q_proj and delete k_up_proj
            setattr(module, "q_proj", new_q_proj)
            delattr(module, "k_up_proj")
            
            # Absorb v_up_proj into o_proj
            v_up_weight = module.v_up_proj.weight.data.clone().reshape(n_heads, head_dim, latent_dim)
            o_weight = module.o_proj.weight.data.clone().reshape(hidden_size, n_heads, head_dim)
            v_up_o = torch.einsum("hdc,Dhd->Dhc", v_up_weight, o_weight)
            
            # Create new linear layer for absorbed o_proj
            new_o_proj_in_features = n_heads * latent_dim
            original_o_proj_bias_exists = hasattr(module.o_proj, 'bias') and module.o_proj.bias is not None
            new_o_proj = torch.nn.Linear(new_o_proj_in_features, hidden_size, bias=original_o_proj_bias_exists)
            new_o_proj = new_o_proj.to(device=target_device, dtype=target_dtype)
            
            new_o_proj.weight.data.copy_(v_up_o.reshape(hidden_size, new_o_proj_in_features).contiguous())
            if original_o_proj_bias_exists:
                new_o_proj.bias.data.copy_(module.o_proj.bias.data)
            
            # Replace module's o_proj and delete v_up_proj
            setattr(module, "o_proj", new_o_proj)
            delattr(module, "v_up_proj")
            
            # Set flag
            setattr(module, "absorb", True)
    
    print(f"Absorption complete. Modified {len(layers_to_modify)} layers.")
    
    return model_copy

def prepare_dataset(dataset, tokenizer, max_length=512, stride=256, batch_size=4):
    """
    Prepare dataset for evaluation by tokenizing and creating overlapping windows.
    Returns a list of batched tensors.
    """
    tokenized_batches = []
    current_batch = []
    
    for i, example in enumerate(tqdm(dataset, desc="Preparing dataset")):
        # Tokenize text
        tokenized = tokenizer(example["text"], return_tensors="pt", truncation=False)
        input_ids = tokenized["input_ids"][0]
        
        # Skip empty examples
        if len(input_ids) <= 1:
            continue
        
        # Create overlapping windows for long sequences
        for start_idx in range(0, max(1, len(input_ids) - 2), stride):
            end_idx = min(start_idx + max_length, len(input_ids))
            window = input_ids[start_idx:end_idx].clone()
            
            # Create a new example with labels = input_ids
            example_dict = {
                "input_ids": window[:-1],  # All tokens except the last one
                "labels": window[1:],      # All tokens except the first one (shifted)
            }
            
            current_batch.append(example_dict)
            
            # When we reach batch_size, add to tokenized_batches and reset
            if len(current_batch) == batch_size:
                tokenized_batches.append(current_batch)
                current_batch = []
    
    # Add any remaining examples
    if current_batch:
        tokenized_batches.append(current_batch)
    
    return tokenized_batches

def evaluate_perplexity(model, batches, device):
    """Evaluate model's perplexity on given batches."""
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_tokens = 0
    batch_losses = []
    
    # Collect per-batch metrics
    batch_perplexities = []
    batch_token_counts = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(batches, desc="Evaluating perplexity")):
            batch_loss = 0.0
            batch_tokens = 0
            
            for example in batch:
                input_ids = example["input_ids"].unsqueeze(0).to(device)
                labels = example["labels"].unsqueeze(0).to(device)
                
                # Forward pass - we need to explicitly pass labels for the loss calculation
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                
                # Count number of tokens and accumulate loss
                num_tokens = labels.numel()
                batch_loss += loss.item() * num_tokens
                batch_tokens += num_tokens
            
            # Calculate mean loss for this batch
            if batch_tokens > 0:
                mean_batch_loss = batch_loss / batch_tokens
                batch_perplexity = math.exp(mean_batch_loss)
                
                # Update totals
                total_loss += batch_loss
                total_tokens += batch_tokens
                
                # Store per-batch metrics
                batch_losses.append(mean_batch_loss)
                batch_perplexities.append(batch_perplexity)
                batch_token_counts.append(batch_tokens)
            
            # Clean up to avoid CUDA OOM errors
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
    
    # Calculate overall metrics
    if total_tokens == 0:
        return {
            "mean_loss": float('nan'),
            "perplexity": float('nan'),
            "batch_losses": [],
            "batch_perplexities": []
        }
    
    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss)
    
    return {
        "mean_loss": mean_loss,
        "perplexity": perplexity,
        "batch_losses": batch_losses,
        "batch_perplexities": batch_perplexities,
        "batch_token_counts": batch_token_counts
    }

def plot_perplexity_comparison(before_results, after_results, title="Perplexity Comparison", save_path=None):
    """Plot perplexity comparison before and after transformation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Overall perplexity comparison
    before_ppl = before_results["perplexity"]
    after_ppl = after_results["perplexity"]
    
    bars = ax1.bar(['Before', 'After'], [before_ppl, after_ppl], color=['blue', 'green'])
    ax1.set_title("Overall Perplexity")
    ax1.set_ylabel("Perplexity (lower is better)")
    
    # Calculate percentage change
    if before_ppl > 0:
        pct_change = (after_ppl - before_ppl) / before_ppl * 100
        ax1.set_xlabel(f"Change: {pct_change:.2f}%")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')
    
    # Plot 2: Per-batch perplexity comparison
    before_batch_ppl = before_results["batch_perplexities"]
    after_batch_ppl = after_results["batch_perplexities"]
    
    # Make sure we use the same number of batches for comparison
    min_batches = min(len(before_batch_ppl), len(after_batch_ppl))
    
    # Only include up to 30 batches in the plot to keep it readable
    plot_batches = min(min_batches, 30)
    
    x = list(range(plot_batches))
    ax2.plot(x, before_batch_ppl[:plot_batches], label='Before', marker='o', color='blue')
    ax2.plot(x, after_batch_ppl[:plot_batches], label='After', marker='s', color='green')
    ax2.set_xlabel("Batch Index")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Per-Batch Perplexity")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    return fig

def main():
    # Configuration
    model_path = "meta-llama/Llama-3.2-1B"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    max_length = 512      # Maximum sequence length for evaluation
    stride = 256          # Stride for overlapping windows
    batch_size = 4        # Batch size for evaluation
    eval_samples = 100    # Number of samples to evaluate (set to None for all)
    
    print(f"Using device: {device}")
    print(f"Loading model: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model with the custom parameters
    print("Loading model and tokenizer...")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        attn_implementation="sdpa",
        partial_rotary_factor=8  # Using custom parameter
    )
    print("Model and tokenizer loaded.")
    
    # Load WikiText dataset
    print("Loading WikiText-2 dataset...")
    # Using validation set as it's smaller and faster to evaluate
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    if eval_samples is not None:
        dataset = dataset.select(range(min(eval_samples, len(dataset))))
    
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Prepare dataset for evaluation
    print("Preparing dataset for evaluation...")
    prepared_batches = prepare_dataset(dataset, tokenizer, max_length, stride, batch_size)
    print(f"Prepared {len(prepared_batches)} batches for evaluation")
    
    # Evaluate original model
    print_separator()
    print("EVALUATING ORIGINAL MODEL")
    start_time = time.time()
    before_results = evaluate_perplexity(model, prepared_batches, device)
    original_eval_time = time.time() - start_time
    print(f"Original model evaluation completed in {original_eval_time:.2f} seconds")
    print(f"Cross-Entropy Loss: {before_results['mean_loss']:.4f}")
    print(f"Perplexity: {before_results['perplexity']:.4f}")
    
    # Apply transformations
    print_separator()
    print("APPLYING TRANSFORMATIONS")
    transformed_model = apply_transformations(model)
    
    # Evaluate transformed model
    print_separator()
    print("EVALUATING TRANSFORMED MODEL")
    start_time = time.time()
    after_results = evaluate_perplexity(transformed_model, prepared_batches, device)
    transformed_eval_time = time.time() - start_time
    print(f"Transformed model evaluation completed in {transformed_eval_time:.2f} seconds")
    print(f"Cross-Entropy Loss: {after_results['mean_loss']:.4f}")
    print(f"Perplexity: {after_results['perplexity']:.4f}")
    
    # Compare results
    print_separator()
    print("COMPARISON OF RESULTS")
    # Calculate percentage changes
    loss_change = (after_results["mean_loss"] - before_results["mean_loss"]) / before_results["mean_loss"] * 100
    ppl_change = (after_results["perplexity"] - before_results["perplexity"]) / before_results["perplexity"] * 100
    speed_change = (original_eval_time - transformed_eval_time) / original_eval_time * 100
    
    print(f"Metric\t\tBefore\t\tAfter\t\tChange (%)")
    print(f"Loss\t\t{before_results['mean_loss']:.4f}\t\t{after_results['mean_loss']:.4f}\t\t{loss_change:+.2f}%")
    print(f"Perplexity\t{before_results['perplexity']:.4f}\t\t{after_results['perplexity']:.4f}\t\t{ppl_change:+.2f}%")
    print(f"Eval Time\t{original_eval_time:.2f}s\t\t{transformed_eval_time:.2f}s\t\t{speed_change:+.2f}%")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        "Metric": ["Loss", "Perplexity", "Evaluation Time (s)"],
        "Before": [before_results["mean_loss"], before_results["perplexity"], original_eval_time],
        "After": [after_results["mean_loss"], after_results["perplexity"], transformed_eval_time],
        "Change (%)": [loss_change, ppl_change, speed_change]
    })
    results_df.to_csv("perplexity_results.csv", index=False)
    print("Results saved to perplexity_results.csv")
    
    # Plot comparison
    plot_perplexity_comparison(before_results, after_results, 
                              title="Model Perplexity Before vs After Transformation",
                              save_path="perplexity_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()