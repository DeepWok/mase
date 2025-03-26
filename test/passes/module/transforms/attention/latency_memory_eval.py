import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from llama.modeling_llama import LlamaForCausalLM
from llama.configuration_llama import LlamaConfig
import time
import gc
import numpy as np
from tqdm import tqdm
import tracemalloc
import os
import pandas as pd
import matplotlib.pyplot as plt
import copy

def print_separator():
    print("=" * 80)

def benchmark_model(model, tokenizer, dataset, device, num_samples=20, max_tokens=100):
    """Benchmark model for inference speed, memory usage, and KV cache size."""
    print_separator()
    print(f"Running benchmark on {num_samples} samples with max_length {max_tokens}")
    
    # Move model to desired device
    model = model.to(device)
    
    # Memory tracking
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    
    # Prepare metrics
    latencies = []
    per_token_latencies = []
    memory_allocated = []
    memory_reserved = []
    kv_cache_sizes = []
    
    # Run inference
    for idx in tqdm(range(min(num_samples, len(dataset)))):
        sample = dataset[idx]['text']
        # Truncate long samples to keep benchmark reasonable
        sample = ' '.join(sample.split()[:50])
        
        # Tokenize
        inputs = tokenizer(sample, return_tensors="pt").to(device)
        
        # Warmup (discard first forward pass)
        if idx == 0:
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=1)
            torch.cuda.empty_cache()
            gc.collect()
        
        # Track memory before generation
        memory_before = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
        # Generate and measure time
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                do_sample=False
            )
        end_time = time.time()
        
        # Track memory after generation
        memory_after = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        memory_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        # Calculate metrics
        input_length = inputs.input_ids.size(1)
        output_length = output.size(1)
        generated_tokens = output_length - input_length
        
        latency = end_time - start_time
        per_token_latency = latency / generated_tokens if generated_tokens > 0 else 0
        
        # Estimate KV cache size based on model architecture
        # For Llama models with standard KV caching
        hidden_size = model.config.hidden_size
        kv_heads = getattr(model.config, 'num_key_value_heads', model.config.num_attention_heads)
        head_dim = hidden_size // model.config.num_attention_heads
        num_layers = model.config.num_hidden_layers
        
        # KV cache size = 2 (K & V) * #layers * #heads * head_dim * seq_len * bytes_per_param
        bytes_per_param = torch.finfo(next(model.parameters()).dtype).bits // 8
        kv_cache = 2 * num_layers * kv_heads * head_dim * output_length * bytes_per_param / (1024 ** 2)  # MB
        
        # Store metrics
        latencies.append(latency)
        per_token_latencies.append(per_token_latency)
        memory_allocated.append(memory_after - memory_before)
        memory_reserved.append(memory_peak)
        kv_cache_sizes.append(kv_cache)
        
        # Clean up for next sample
        torch.cuda.empty_cache()
        gc.collect()
    
    # Calculate final metrics
    results = {
        "avg_latency_s": np.mean(latencies),
        "avg_tokens_per_second": 1.0 / np.mean(per_token_latencies),
        "avg_per_token_latency_ms": np.mean(per_token_latencies) * 1000,
        "avg_memory_allocated_mb": np.mean(memory_allocated),
        "peak_memory_mb": np.max(memory_reserved),
        "avg_kv_cache_size_mb": np.mean(kv_cache_sizes),
    }
    
    # Stop memory tracking
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["python_memory_mb"] = current / (1024 * 1024)
    
    # Print results
    print("\nBenchmark Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")
    
    return results

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
                    if modified_layers == 1:
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
                if modified_layers == 1:
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
                S_sqrtV = torch.einsum('hd, hdD->hdD', S_sqrt, V.mH)  # Note the einsum pattern from your updated code
                
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
            if idx == 0:
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

def test_generation(model, tokenizer, device):
    """Test generation to ensure the model works correctly."""
    prompt = "Tell me a story"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    
    generated_text = tokenizer.batch_decode(output)[0]
    print(f"\nGeneration test:\n{generated_text}\n")

def plot_comparison(before_results, after_results, figure_title, save_path=None):
    """Plot comparison of benchmark results before and after transformation."""
    metrics = {
        "Tokens per Second": [before_results["avg_tokens_per_second"], after_results["avg_tokens_per_second"]],
        "Latency per Token (ms)": [before_results["avg_per_token_latency_ms"], after_results["avg_per_token_latency_ms"]],
        "Memory Usage (MB)": [before_results["avg_memory_allocated_mb"], after_results["avg_memory_allocated_mb"]],
        "KV Cache Size (MB)": [before_results["avg_kv_cache_size_mb"], after_results["avg_kv_cache_size_mb"]]
    }
    
    # Calculate improvement percentages
    improvements = {}
    for metric, values in metrics.items():
        if metric == "Latency per Token (ms)":
            # For latency, lower is better
            improvement = (values[0] - values[1]) / values[0] * 100 if values[0] > 0 else 0
        else:
            # For other metrics, higher is better
            improvement = (values[1] - values[0]) / values[0] * 100 if values[0] > 0 else 0
        improvements[metric] = improvement
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(figure_title, fontsize=16)
    
    # Plot each metric
    for i, (metric, values) in enumerate(metrics.items()):
        row, col = i // 2, i % 2
        ax = axs[row, col]
        
        bars = ax.bar(['Before', 'After'], values, color=['blue', 'green'])
        ax.set_title(f"{metric}\nImprovement: {improvements[metric]:.2f}%")
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    return fig

def main():
    # Configuration
    model_path = "meta-llama/Llama-3.2-1B"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_samples = 20
    max_tokens = 100
    
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
        partial_rotary_factor=8  # Using your custom parameter
    )
    print("Model and tokenizer loaded.")
    
    # Load WikiText dataset
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Verify generation works before transformations
    print("Testing generation before transformations...")
    test_generation(model, tokenizer, device)
    
    # Benchmark before transformation
    print_separator()
    print("BENCHMARK BEFORE TRANSFORMATION")
    before_results = benchmark_model(model, tokenizer, dataset, device, num_samples, max_tokens)
    
    # Apply transformations
    print_separator()
    print("APPLYING TRANSFORMATIONS")
    transformed_model = apply_transformations(model)
    
    # Verify generation works after transformations
    print("Testing generation after transformations...")
    test_generation(transformed_model, tokenizer, device)
    
    # Benchmark after transformation
    print_separator()
    print("BENCHMARK AFTER TRANSFORMATION")
    after_results = benchmark_model(transformed_model, tokenizer, dataset, device, num_samples, max_tokens)
    
    # Compare results
    print_separator()
    print("COMPARISON OF RESULTS")
    print("Metric\t\t\tBefore\t\tAfter\t\tChange (%)")
    for metric in ["avg_tokens_per_second", "avg_per_token_latency_ms", "avg_memory_allocated_mb", "avg_kv_cache_size_mb"]:
        before = before_results[metric]
        after = after_results[metric]
        if metric == "avg_per_token_latency_ms":
            # For latency, lower is better
            change = (before - after) / before * 100 if before > 0 else 0
        else:
            # For other metrics, higher is better
            change = (after - before) / before * 100 if before > 0 else 0
        print(f"{metric:25s}\t{before:.4f}\t{after:.4f}\t{change:+.2f}%")
    
    # Save full results to CSV
    results_df = pd.DataFrame({
        "Metric": list(before_results.keys()),
        "Before": list(before_results.values()),
        "After": list(after_results.values())
    })
    results_df.to_csv("benchmark_results.csv", index=False)
    print("Results saved to benchmark_results.csv")
    
    # Plot comparison
    fig = plot_comparison(before_results, after_results, "Model Performance Before vs After Transformation", "benchmark_results.png")
    plt.show()

if __name__ == "__main__":
    main()