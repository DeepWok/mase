from transformers import AutoTokenizer
from llama.modeling_llama import LlamaForCausalLM
from llama.configuration_llama import LlamaConfig
import torch
from tqdm import tqdm

# --- Stage 1: Load Model ---
# Using Meta's Llama-3.2-1B from Hugging Face
model_path = "meta-llama/Llama-3.2-1B"
# NOTE: Choose the appropriate device, e.g., 'cuda:0', 'cpu', or 'auto'
device_map_setting = 'auto'
# NOTE: Specify the desired CUDA device if generating on GPU, e.g., "cuda:0"
generation_device = "cuda:0"

print("Loading model and tokenizer...")
# Using AutoTokenizer and AutoModelForCausalLM instead of specific Llama classes
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path,
    device_map=device_map_setting,
    attn_implementation="sdpa",
    partial_rotary_factor=8
)
print(model)
print("Model and tokenizer loaded.")

# Model configuration details
hidden_size = model.config.hidden_size
n_heads = model.config.num_attention_heads
kv_heads = model.config.num_key_value_heads
head_dim = model.config.hidden_size // model.config.num_attention_heads
latent_dim = kv_heads * head_dim
kv_groups = model.config.num_attention_heads // model.config.num_key_value_heads

# --- Stage 2: Initial Weight Modification (Identity Matrices) ---
print("\n--- Applying Initial Weight Modification (Identity) ---")
with torch.no_grad():
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            target_device = module.weight.data.device
            target_dtype = module.weight.data.dtype
            if 'k_up_proj' in name or "v_up_proj" in name:
                # Constructing identity matrix based on dimensions
                identity_weight = torch.stack(
                    [torch.eye(latent_dim).reshape(kv_heads, head_dim, latent_dim)] * kv_groups,
                    dim=1
                ).reshape(hidden_size, latent_dim).contiguous().to(target_device, target_dtype)
                if 'k_up_proj' in name:
                    # Reshape/transpose specific to k_up_proj
                    identity_weight = identity_weight.view(hidden_size, kv_heads, head_dim).transpose(1, 2).contiguous().view(hidden_size, latent_dim)
                module.weight.data.copy_(identity_weight) # Use copy_ for in-place update
            elif 'k_proj' in name: # Apply reshaping to k_proj weights and bias
                # Reshape weight
                reshaped_weight = module.weight.data.view(kv_heads, head_dim, hidden_size).transpose(0, 1).contiguous().view(latent_dim, hidden_size)
                module.weight.data.copy_(reshaped_weight)
                # Reshape bias if it exists
                if hasattr(module, 'bias') and module.bias is not None:
                    reshaped_bias = module.bias.data.view(kv_heads, head_dim).transpose(0, 1).contiguous().view(latent_dim)
                    module.bias.data.copy_(reshaped_bias)
print("Initial modification complete.")

# --- Stage 3: First Generation ---
print("\n--- Performing First Generation ---")
prompt = "Tell me a story"
inputs = tokenizer(prompt, return_tensors="pt").to(generation_device)
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print("Generated text (1):")
print(tokenizer.batch_decode(output)[0])

# --- Stage 4: Second Weight Modification (Orthogonalization via SVD) ---
print("\n--- Applying Second Weight Modification (Orthogonalization) ---")
with torch.no_grad():
    for name, module in model.named_modules():
        # Check if the module is a self-attention layer
        if isinstance(module, torch.nn.Module) and "self_attn" in name and hasattr(module, 'k_up_proj'):
            target_device = module.q_proj.weight.device
            target_dtype = module.q_proj.weight.dtype
            
            # Orthogonalize q_proj and k_up_proj
            k_up_weight = module.k_up_proj.weight.data.clone().reshape(n_heads, head_dim, latent_dim)
            q_weight = module.q_proj.weight.data.clone().reshape(n_heads, head_dim, hidden_size)
            if module.q_proj.bias is not None:
                q_bias = module.q_proj.bias.data.clone().reshape(n_heads, head_dim, 1)
                q_weight = torch.cat([q_weight, q_bias], dim=-1) # Append bias as a column
            
            q_k_up = torch.einsum("hdc,hdD->hcD", k_up_weight, q_weight)
            
            # SVD - Use torch.linalg.svd for stability
            U, S, Vh = torch.linalg.svd(q_k_up, full_matrices=False)
            V = Vh.mH # Conjugate transpose for V
            
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
                S_sqrtV_weights = S_sqrtV[:, :, :-1] # Separate weights from bias column
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
print("Orthogonalization complete.")

# --- Stage 5: Second Generation ---
print("\n--- Performing Second Generation ---")
inputs = tokenizer(prompt, return_tensors="pt").to(generation_device)
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print("Generated text (2):")
print(tokenizer.batch_decode(output)[0])

# --- Stage 6: Third Weight Modification (Absorption) ---
print("\n--- Applying Third Weight Modification (Absorption) ---")
with torch.no_grad():
    layers_to_modify = []
    # First, identify layers that still need absorption
    for name, module in model.named_modules():
        if "self_attn" in name and hasattr(module, 'k_up_proj') and hasattr(module, 'v_up_proj'):
            layers_to_modify.append((name, module))
    
    # Now, modify them. This avoids issues with modifying modules while iterating.
    for name, module in layers_to_modify:
        target_device = module.q_proj.weight.device
        target_dtype = module.q_proj.weight.dtype
        
        # Absorb k_up_proj into q_proj
        k_up_weight = module.k_up_proj.weight.data.clone().reshape(n_heads, head_dim, latent_dim)
        q_weight = module.q_proj.weight.data.clone().reshape(n_heads, head_dim, hidden_size)
        q_bias_data = None
        if module.q_proj.bias is not None:
            q_bias = module.q_proj.bias.data.clone().reshape(n_heads, head_dim, 1)
            q_weight = torch.cat([q_weight, q_bias], dim=-1) # Append bias column
        
        q_k_up = torch.einsum("hdc,hdD->hcD", k_up_weight, q_weight)
        
        # Create new linear layer for absorbed q_proj
        new_q_proj_out_features = n_heads * latent_dim
        new_q_proj = torch.nn.Linear(hidden_size, new_q_proj_out_features, bias=(module.q_proj.bias is not None))
        new_q_proj = new_q_proj.to(device=target_device, dtype=target_dtype)
        
        if module.q_proj.bias is not None:
            new_q_proj.bias.data.copy_(q_k_up[:, :, -1].reshape(-1).contiguous())
            q_k_up_weights = q_k_up[:, :, :-1] # Separate weights
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
print("Absorption complete.")

# --- Stage 7: Third Generation ---
print("\n--- Performing Third Generation ---")
inputs = tokenizer(prompt, return_tensors="pt").to(generation_device)
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print("Generated text (3):")
print(tokenizer.batch_decode(output)[0])
print("\nScript finished.")

# -----------------------------------------------
# Alternative approach using pipeline (commented out)
# This is a simpler approach shown in the tutorial
# but doesn't allow for the weight modifications
# -----------------------------------------------
'''
# Import pipeline
from transformers import pipeline

# Create a text-generation pipeline
pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")

# Generate text
result = pipe("Tell me a story", max_new_tokens=50)
print(result[0]['generated_text'])
'''