#!/usr/bin/env python3
# Simple script to test MLA transformation with direct module-level fixes

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import os
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
CHECKPOINT = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
OUTPUT_DIR = "./results_mla_test"
NUM_TRAIN_EPOCHS = 1
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 128
TRAIN_SUBSET_SIZE = 100
INFERENCE_TEXT = "Explain the concept of artificial intelligence in simple terms: "

# Fix for the rotary embedding application in MLA module
def fix_mla_apply_rotary_emb():
    """Apply fixes to the apply_rotary_emb function in MLA module"""
    try:
        # Import the MLA module
        from chop.nn.attention.modules.mla import apply_rotary_emb
        
        # Define a fixed version that properly handles tensor shapes
        def fixed_apply_rotary_emb(x, freqs_cis):
            # Original rotary embedding implementation
            # But add extra checks and reshaping to ensure tensor compatibility
            
            # Get the original shape of x
            bsz, seqlen, n_heads, head_dim = x.shape
            
            # Debug logging
            logger.debug(f"apply_rotary_emb shapes - x: {x.shape}, freqs_cis: {freqs_cis.shape}")
            
            # Ensure freqs_cis has the right shape for applying rotary embeddings
            if freqs_cis.dim() == 2:  # (seq_len, dim/2)
                # Handle the standard case where freqs_cis is [seq_len, dim/2]
                # But make sure seq_len matches
                if freqs_cis.size(0) != seqlen:
                    logger.warning(f"Mismatch in sequence lengths - x: {seqlen}, freqs_cis: {freqs_cis.size(0)}")
                    # Resize freqs_cis to match x's sequence length
                    if freqs_cis.size(0) > seqlen:
                        freqs_cis = freqs_cis[:seqlen]
                    else:
                        # Repeat the last position for extra length
                        pad_len = seqlen - freqs_cis.size(0)
                        freqs_cis = torch.cat([freqs_cis, freqs_cis[-1:].repeat(pad_len, 1)], dim=0)
                
                # Check if head_dim is compatible with freqs_cis
                if freqs_cis.size(1) * 2 != head_dim:
                    logger.warning(
                        f"Dimension mismatch - head_dim: {head_dim}, freqs_cis dim: {freqs_cis.size(1)*2}. "
                        f"Adjusting freqs_cis."
                    )
                    # Adjust freqs_cis to match head_dim
                    required_dim = head_dim // 2
                    if freqs_cis.size(1) > required_dim:
                        # Truncate if too large
                        freqs_cis = freqs_cis[:, :required_dim]
                    else:
                        # Repeat if too small
                        repeat_factor = math.ceil(required_dim / freqs_cis.size(1))
                        freqs_cis = torch.cat([freqs_cis] * repeat_factor, dim=1)[:, :required_dim]
                
                # Reshape to proper dimensions for the calculation
                freqs_cis = freqs_cis.view(1, seqlen, 1, -1)
            
            # Reshape x for applying rotary embeddings
            if head_dim % 2 != 0:
                logger.warning(f"Head dimension {head_dim} is not even, padding for rotary embeddings")
                head_dim_padded = head_dim + 1
                x_padded = torch.zeros(
                    bsz, seqlen, n_heads, head_dim_padded, device=x.device, dtype=x.dtype
                )
                x_padded[:, :, :, :head_dim] = x
                x = x_padded
            
            # Split the last dimension into real and imaginary parts
            x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
            
            # Apply rotary embeddings
            freqs_cis = freqs_cis.to(device=x.device, dtype=torch.float32)
            freqs_complex = torch.view_as_complex(freqs_cis.reshape(*freqs_cis.shape[:-1], -1, 2))
            
            # Multiply by complex exponential
            x_rotated = x_complex * freqs_complex
            
            # Convert back to real
            x_out = torch.view_as_real(x_rotated).flatten(-2)
            
            # Restore original shape
            if head_dim % 2 != 0:
                x_out = x_out[:, :, :, :head_dim]
            
            return x_out
        
        # Apply the fix
        from chop.nn.attention.modules import mla
        mla.apply_rotary_emb = fixed_apply_rotary_emb
        
        logger.info("Successfully patched MLA apply_rotary_emb function")
        return True
    
    except Exception as e:
        logger.error(f"Error patching MLA apply_rotary_emb: {e}")
        return False

# Fix the MLA forward method to handle different tensor shapes
def fix_mla_forward():
    """Apply fixes to the MLA forward method"""
    try:
        # Import the MLA module
        from chop.nn.attention.modules.mla import MLA
        
        # Store original forward
        original_forward = MLA.forward
        
        # Define fixed forward method
        def fixed_forward(self, x, start_pos=0, freqs_cis=None, mask=None):
            """
            Fixed MLA forward method with better tensor shape handling
            """
            bsz, seqlen, _ = x.shape
            
            # Ensure proper head dimensions
            assert self.dim % self.n_heads == 0, f"Dimension {self.dim} not divisible by n_heads {self.n_heads}"
            
            # Debug logging
            logger.debug(f"MLA input shapes - x: {x.shape}, freqs_cis: {None if freqs_cis is None else freqs_cis.shape}")
            
            # Get query matrix
            q = self.wq(x)
            
            # Reshape for KV projection
            original_shape = q.shape
            
            # Apply rotary embeddings to query if provided
            if self.qk_rope_head_dim > 0:
                if freqs_cis is None:
                    logger.warning("freqs_cis not provided for rotary embeddings, using random values")
                    # Generate random embeddings
                    dim = self.qk_rope_head_dim
                    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) / dim))
                    t = torch.arange(seqlen, dtype=torch.float32, device=x.device)
                    freqs = torch.outer(t, freqs)
                    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
                
                # Apply rotary position embeddings
                q_pe = self.reshape_for_broadcast(q, bsz, seqlen, self.qk_rope_head_dim)
                
                # Fix freqs_cis shape to match q_pe
                if freqs_cis.shape[0] != seqlen:
                    logger.warning(f"Adjusting freqs_cis from shape {freqs_cis.shape} to match sequence length {seqlen}")
                    if freqs_cis.shape[0] > seqlen:
                        freqs_cis = freqs_cis[:seqlen]
                    else:
                        # Pad by repeating
                        pad_len = seqlen - freqs_cis.shape[0]
                        freqs_cis = torch.cat([freqs_cis, freqs_cis[-1:].repeat(pad_len, 1)], dim=0)
                
                # Apply rotary embeddings
                q_pe = self.apply_rotary_emb(q_pe, freqs_cis)
                
                # Reshape back
                q = self.reshape_from_broadcast(q_pe, original_shape)
            
            # Process the KV path with low-rank layers
            kv_input = self.wkv_a(x)
            kv = self.wkv_b(kv_input)
            
            # Normalize if needed
            if self.kv_norm is not None:
                kv = self.kv_norm(kv)
            
            # Reshape KV for splitting K and V
            kv = kv.reshape(bsz, seqlen, 2, self.v_head_dim, self.n_heads)
            k, v = kv[:, :, 0], kv[:, :, 1]
            
            # Compute attention scores
            k = k.transpose(2, 3)
            q = q.reshape(bsz, seqlen, self.n_heads, self.v_head_dim)
            
            # Safe transpose and reshape
            # These operations ensure dimensions are as expected
            q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
            v = v.transpose(2, 3).transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
            
            # Fix tensor shapes for matmul if needed
            if q.size(-1) != k.size(-2):
                logger.warning(f"Fixing dimension mismatch: q[-1]={q.size(-1)}, k[-2]={k.size(-2)}")
                # Adjust the smaller dimension to match the larger one
                if q.size(-1) < k.size(-2):
                    q_new = torch.zeros(
                        *q.shape[:-1], k.size(-2), device=q.device, dtype=q.dtype
                    )
                    q_new[..., :q.size(-1)] = q
                    q = q_new
                else:
                    k_new = torch.zeros(
                        *k.shape[:-2], q.size(-1), k.size(-1), device=k.device, dtype=k.dtype
                    )
                    k_new[..., :k.size(-2), :] = k
                    k = k_new
            
            # Compute attention
            scores = torch.matmul(q, k)  # (bsz, n_heads, seqlen, seqlen)
            scores = scores / math.sqrt(self.v_head_dim)
            
            # Apply mask if provided
            if mask is not None:
                scores = scores + mask
            
            # Apply attention
            attn_weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            output = torch.matmul(attn_weights, v)  # (bsz, n_heads, seqlen, head_dim)
            
            # Reshape output back to original dimensions
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            
            # Final projection
            return self.wo(output)
        
        # Apply the fix
        MLA.forward = fixed_forward
        
        # Also fix apply_rotary_emb in the MLA class
        MLA.apply_rotary_emb = fix_mla_apply_rotary_emb
        
        logger.info("Successfully patched MLA forward method")
        return True
    
    except Exception as e:
        logger.error(f"Error patching MLA forward: {e}")
        return False

# Fix MLAAttentionWrapper
def fix_mla_wrapper():
    """Fix the MLAAttentionWrapper class"""
    try:
        # Import the wrapper class
        from chop.passes.module.transforms.attention.attention_transform_helper import MLAAttentionWrapper
        
        # Save original methods
        original_init = MLAAttentionWrapper.__init__
        original_forward = MLAAttentionWrapper.forward
        
        # Define fixed methods
        def fixed_init(self, mla_module):
            """Fixed initialization to avoid buffer duplication"""
            torch.nn.Module.__init__(self)
            
            self.mla = mla_module
            self.is_mla_wrapper = True
            
            # Get dimensions
            self.hidden_size = mla_module.dim
            self.num_heads = mla_module.n_heads
            
            # Position counter for tracking
            self.register_buffer('position_counter', torch.zeros(1, dtype=torch.int), persistent=False)
            
            # Import precompute_freqs_cis
            from chop.nn.attention.modules.mla import precompute_freqs_cis
            
            # Precompute frequencies
            self.register_buffer(
                'freqs_cis',
                precompute_freqs_cis(mla_module.model_args),
                persistent=False
            )
        
        def fixed_forward(self, hidden_states, attention_mask=None, position_ids=None, 
                         past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
            """Fixed forward method with better handling of tensors and caching"""
            batch_size, seq_len = hidden_states.size()[:2]
            
            # Get dtype from MLA
            param = next(self.mla.parameters(), None)
            target_dtype = param.dtype if param is not None else torch.bfloat16
            device = hidden_states.device
            
            # Convert inputs
            hidden_states = hidden_states.to(target_dtype)
            
            # Process mask
            mla_mask = None
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    mla_mask = attention_mask.squeeze(1)
                else:
                    mla_mask = attention_mask
                
                if mla_mask.dtype != torch.bool:
                    mla_mask = mla_mask.to(target_dtype)
            
            # Handle start position
            start_pos = 0
            if past_key_value is not None:
                if isinstance(past_key_value, tuple) and len(past_key_value) == 2:
                    start_pos = past_key_value[0].size(2)
                elif hasattr(past_key_value, "get_seq_length"):
                    start_pos = past_key_value.get_seq_length()
                elif hasattr(self, 'position_counter'):
                    start_pos = self.position_counter.item()
                    self.position_counter += seq_len
            
            # Process position embeddings
            freqs_cis = self.freqs_cis
            if position_ids is not None:
                # Handle batched position_ids
                if position_ids.dim() > 1:
                    position_ids_flat = position_ids[0]
                    freqs_cis = torch.index_select(self.freqs_cis, 0, position_ids_flat)
                else:
                    freqs_cis = torch.index_select(self.freqs_cis, 0, position_ids)
            else:
                # Use sequential positions
                if start_pos + seq_len <= freqs_cis.size(0):
                    freqs_cis = freqs_cis[start_pos:start_pos+seq_len]
                else:
                    # Handle case where sequence length exceeds available positions
                    logger.warning(f"Sequence position {start_pos + seq_len} exceeds max length {freqs_cis.size(0)}")
                    available = freqs_cis[start_pos:] if start_pos < freqs_cis.size(0) else freqs_cis[-1:].expand(1, -1)
                    freqs_cis = available.repeat((seq_len + available.size(0) - 1) // available.size(0) + 1, 1)[:seq_len]
            
            # Ensure on right device
            freqs_cis = freqs_cis.to(device=device)
            
            # Debug logs
            logger.debug(f"MLAWrapper forward shapes - hidden: {hidden_states.shape}, freqs_cis: {freqs_cis.shape}")
            logger.debug(f"MLAWrapper forward mask shape: {None if mla_mask is None else mla_mask.shape}")
            
            # Call MLA forward
            try:
                output = self.mla(
                    x=hidden_states,
                    start_pos=start_pos,
                    freqs_cis=freqs_cis,
                    mask=mla_mask
                )
            except Exception as e:
                logger.error(f"Error in MLA forward: {e}")
                logger.error(f"  hidden_states: {hidden_states.shape}")
                logger.error(f"  start_pos: {start_pos}")
                logger.error(f"  freqs_cis: {freqs_cis.shape}")
                logger.error(f"  mla_mask: {None if mla_mask is None else mla_mask.shape}")
                raise
            
            # Match original dtype
            orig_dtype = kwargs.get('input_dtype', hidden_states.dtype)
            if output.dtype != orig_dtype:
                output = output.to(orig_dtype)
            
            # Handle caching
            present_key_value = None
            if use_cache:
                head_dim = self.hidden_size // self.num_heads
                dummy_key = torch.zeros(
                    (batch_size, self.num_heads, seq_len, head_dim),
                    device=device, dtype=orig_dtype
                )
                dummy_value = torch.zeros_like(dummy_key)
                present_key_value = (dummy_key, dummy_value)
            
            return output, None, present_key_value
        
        # Apply fixes
        MLAAttentionWrapper.__init__ = fixed_init
        MLAAttentionWrapper.forward = fixed_forward
        
        logger.info("Successfully patched MLAAttentionWrapper")
        return True
    
    except Exception as e:
        logger.error(f"Error patching MLAAttentionWrapper: {e}")
        return False

# Fix the DynamicCache.from_legacy_cache method
def fix_dynamic_cache():
    """Patch transformers.generation.utils.DynamicCache.from_legacy_cache"""
    try:
        from transformers.generation import utils as gen_utils
        
        # Save original
        original_from_legacy_cache = gen_utils.DynamicCache.from_legacy_cache
        
        # Define fixed method
        @classmethod
        def fixed_from_legacy_cache(cls, past_key_values):
            # Handle various formats and issues
            
            # If None, return empty cache
            if past_key_values is None:
                return cls()
            
            # Handle tensor format issues
            for layer_idx in range(len(past_key_values)):
                if past_key_values[layer_idx] is None:
                    # Create empty tuple
                    past_key_values[layer_idx] = (None, None)
                elif isinstance(past_key_values[layer_idx], torch.Tensor):
                    # Single tensor case
                    tensor = past_key_values[layer_idx]
                    past_key_values[layer_idx] = (tensor, torch.zeros_like(tensor))
                elif (isinstance(past_key_values[layer_idx], tuple) and 
                      len(past_key_values[layer_idx]) == 1):
                    # Tuple with one element
                    key = past_key_values[layer_idx][0]
                    value = torch.zeros_like(key) if key is not None else None
                    past_key_values[layer_idx] = (key, value)
            
            # Call original
            return original_from_legacy_cache(past_key_values)
        
        # Apply patch
        gen_utils.DynamicCache.from_legacy_cache = fixed_from_legacy_cache
        
        logger.info("Successfully patched DynamicCache.from_legacy_cache")
        return True
    
    except Exception as e:
        logger.error(f"Error patching DynamicCache: {e}")
        return False

# Add fix for mla_by_type
def fix_mla_by_type():
    """Fix the mla_by_type function"""
    try:
        from chop.passes.module.transforms.attention import attention
        
        # Store original
        original_mla_by_type = attention.mla_by_type
        
        # Define fixed function
        def fixed_mla_by_type(network, pass_args):
            # Handle boolean configs
            fixed_args = {}
            for type_name, config_entry in pass_args.items():
                # Skip internal keys
                if type_name in ["by", "verbose"]:
                    fixed_args[type_name] = config_entry
                    continue
                
                # Handle boolean values
                if isinstance(config_entry, bool):
                    fixed_args[type_name] = {"config": {"name": "mla"}}
                elif isinstance(config_entry, dict):
                    if "config" in config_entry and isinstance(config_entry["config"], bool):
                        # Fix boolean config
                        new_entry = dict(config_entry)
                        new_entry["config"] = {"name": "mla"}
                        fixed_args[type_name] = new_entry
                    else:
                        # Keep as is
                        fixed_args[type_name] = config_entry
                else:
                    # Keep as is
                    fixed_args[type_name] = config_entry
            
            # Call original with fixed args
            return original_mla_by_type(network, fixed_args)
        
        # Apply fix
        attention.mla_by_type = fixed_mla_by_type
        
        logger.info("Successfully patched mla_by_type")
        return True
    
    except Exception as e:
        logger.error(f"Error patching mla_by_type: {e}")
        return False

def create_dataset(tokenizer):
    """Create a simple dataset for testing"""
    # Load a small subset
    raw_dataset = load_dataset(
        DATASET_NAME, 
        DATASET_CONFIG, 
        split=f"train[:{TRAIN_SUBSET_SIZE}]"
    )
    
    # Simple tokenization with padding
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors="np"
        )
    
    # Process dataset
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    # Add labels
    def add_labels(example):
        example["labels"] = example["input_ids"].copy()
        return example
    
    final_dataset = tokenized_dataset.map(add_labels)
    
    # Split dataset
    split_dataset = final_dataset.train_test_split(test_size=0.1)
    
    return split_dataset

def finetune_model(model, tokenizer, dataset, output_dir):
    """Run fine-tuning on the model"""
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="no",
        evaluation_strategy="no",
        fp16=False,
        bf16=False,
        gradient_accumulation_steps=4,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()
    
    return model

def run_inference(model, tokenizer, text):
    """Run a simple inference test"""
    model.eval()
    device = model.device
    
    # Prepare input
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Generate with fallbacks
    with torch.no_grad():
        try:
            # First try with use_cache=True
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        except Exception as e:
            logger.warning(f"Generation with cache failed: {e}")
            try:
                # Then try with use_cache=False
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False
                )
            except Exception as e2:
                logger.error(f"Generation without cache also failed: {e2}")
                # Last resort: direct forward pass
                outputs = model(**inputs).logits.argmax(dim=-1)
    
    # Decode output
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Apply all fixes
    logger.info("\n--- Applying MLA module fixes ---")
    mla_forward_fixed = fix_mla_forward()
    mla_wrapper_fixed = fix_mla_wrapper()
    dynamic_cache_fixed = fix_dynamic_cache()
    mla_by_type_fixed = fix_mla_by_type()
    
    if not all([mla_forward_fixed, mla_wrapper_fixed, dynamic_cache_fixed, mla_by_type_fixed]):
        logger.warning("Some fixes could not be applied - proceed with caution")
    
    # Load model
    logger.info(f"\n--- Loading model from {CHECKPOINT} ---")
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    
    # Handle pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Move to device
    model.to(device)
    
    # Save original model
    import copy
    original_model = copy.deepcopy(model)
    
    # Test original model
    logger.info("\n--- Testing original model ---")
    original_output = run_inference(original_model, tokenizer, INFERENCE_TEXT)
    logger.info(f"Original model output:\n{original_output}")
    
    # Create dataset
    logger.info("\n--- Creating dataset ---")
    dataset = create_dataset(tokenizer)
    logger.info(f"Dataset created with {len(dataset['train'])} training examples")
    
    # Fine-tune original model
    logger.info("\n--- Fine-tuning original model ---")
    original_output_dir = f"{OUTPUT_DIR}/original"
    finetune_model(original_model, tokenizer, dataset, original_output_dir)
    
    # Test fine-tuned original model
    logger.info("\n--- Testing fine-tuned original model ---")
    finetuned_output = run_inference(original_model, tokenizer, INFERENCE_TEXT)
    logger.info(f"Fine-tuned original model output:\n{finetuned_output}")
    
    # Transform model to MLA
    logger.info("\n--- Transforming model to MLA ---")
    from chop.passes.module.transforms import attention_transform_pass
    
    # Move to CPU for transformation
    model = model.cpu()
    
    # Configure transformation
    transform_args = {
        "by": "type",
        "llama": {
            "config": {
                "name": "mla",
                "max_seq_len": MAX_SEQ_LENGTH,
                "max_batch_size": BATCH_SIZE
            }
        },
        "verbose": True
    }
    
    # Run transformation
    try:
        mla_model, stats = attention_transform_pass(model, transform_args)
        logger.info(f"Transformation stats: {stats}")
        
        # Verify transformation
        found_mla = False
        for name, module in mla_model.named_modules():
            if 'MLA' in type(module).__name__ or hasattr(module, 'is_mla_wrapper'):
                found_mla = True
                logger.info(f"Found MLA module: {name}")
        
        if not found_mla:
            logger.warning("No MLA modules found after transformation!")
        
        # Move back to device
        mla_model = mla_model.to(device)
    except Exception as e:
        logger.error(f"Error during transformation: {e}")
        mla_model = original_model
        logger.warning("Using original model due to transformation error")
    
    # Test MLA model
    logger.info("\n--- Testing MLA model (pre-finetune) ---")
    # Disable use_cache if needed
    if hasattr(mla_model.config, "use_cache"):
        orig_use_cache = mla_model.config.use_cache
        mla_model.config.use_cache = False
    
    mla_output = run_inference(mla_model, tokenizer, INFERENCE_TEXT)
    logger.info(f"MLA model output (pre-finetune):\n{mla_output}")
    
    # Fine-tune MLA model
    logger.info("\n--- Fine-tuning MLA model ---")
    try:
        mla_output_dir = f"{OUTPUT_DIR}/mla"
        finetune_model(mla_model, tokenizer, dataset, mla_output_dir)
    except Exception as e:
        logger.error(f"Error during MLA fine-tuning: {e}")
    
    # Test fine-tuned MLA model
    logger.info("\n--- Testing fine-tuned MLA model ---")
    finetuned_mla_output = run_inference(mla_model, tokenizer, INFERENCE_TEXT)
    logger.info(f"Fine-tuned MLA model output:\n{finetuned_mla_output}")
    
    # Restore use_cache setting
    if hasattr(mla_model.config, "use_cache") and 'orig_use_cache' in locals():
        mla_model.config.use_cache = orig_use_cache
    
    # Comparison
    logger.info("\n--- Results comparison ---")
    logger.info(f"Original model:\n{original_output}\n")
    logger.info(f"Fine-tuned original model:\n{finetuned_output}\n")
    logger.info(f"MLA model (pre-finetune):\n{mla_output}\n")
    logger.info(f"MLA model (post-finetune):\n{finetuned_mla_output}\n")

if __name__ == "__main__":
    main()



# import torch
# from transformers.models.bert.modeling_bert import(
#     BertSelfAttention, 
#     BertSdpaSelfAttention, 
#     BertSelfOutput, 
#     BertAttention
# )
# from transformers.models.gpt2.modeling_gpt2 import (
#     GPT2SdpaAttention,
#     GPT2Block,
# )
# from chop.nn.attention.modules import attention_module_map
# from ...module_modify_helper import replace_by_name, instantiate_module
# from ...state_dict_map import match_a_pattern, check_is_huggingface_model
# from .attention_transform_helper import MLAWrapper, llama2_to_mla_init, transform_llama2_to_mla

# from transformers.models.llama.modeling_llama import (
#     LlamaAttention,
#     LlamaDecoderLayer
# )

# def get_config(config: dict, name: str):
#     if name in config:
#         return config[name]["config"]
#     else:
#         return config["default"]["config"]


# def mla_by_type(network, pass_args):
#     # Import the necessary classes
#     from transformers.models.llama.modeling_llama import LlamaAttention
#     from .attention_transform_helper import MLAAttentionWrapper, transform_llama2_to_mla, llama2_to_mla_init
    
#     transformed_count = 0
#     stats = {}  # Create stats dictionary to return
    
#     for type_name, config_entry in pass_args.items():
#         # Collect all modules
#         n_m = {}
#         for n, m in network.named_modules():
#             n_m[n] = m
            
#         # Extract config - handle both nested and flat config formats
#         if "config" in config_entry:
#             config = config_entry["config"].copy()
#         else:
#             config = config_entry.copy() 
            
#         # Get postfix, defaulting to "mla" if not specified
#         postfix = config.pop("name", "mla")
#         print(f"Using postfix: {postfix}, config: {config}")
        
#         stats[type_name] = {"transformed_modules": []}  # Track transformed modules in stats
            
#         if type_name == "llama":
#             print(f"Looking for Llama attention modules...")
            
#             # Find and transform all matching modules
#             for n, m in n_m.items():
#                 # Check if it's an attention module (by any detection method)
#                 is_attention = False
#                 if isinstance(m, LlamaAttention):
#                     is_attention = True
#                     print(f"Found exact match: {n}")
#                 elif "Attention" in type(m).__name__ and "Llama" in type(m).__name__:
#                     is_attention = True
#                     print(f"Found name match: {n}")
                    
#                 if is_attention:
#                     try:
#                         # Create MLA module (using helper functions directly)
#                         mla_module = llama2_to_mla_init(m, {"config": config})
                        
#                         # Transform weights
#                         mla_module = transform_llama2_to_mla(m, mla_module)
                        
#                         # Create wrapper with proper interface 
#                         wrapped_module = MLAAttentionWrapper(mla_module)
                        
#                         # Replace in model - find parent module and set attribute
#                         if '.' in n:
#                             parent_name, child_name = n.rsplit('.', 1)
#                             parent = network
#                             for part in parent_name.split('.'):
#                                 parent = getattr(parent, part)
#                             setattr(parent, child_name, wrapped_module)
#                         else:
#                             setattr(network, n, wrapped_module)
                            
#                         transformed_count += 1
#                         stats[type_name]["transformed_modules"].append(n)
#                         print(f"Successfully transformed {n}")
#                     except Exception as e:
#                         print(f"Error transforming {n}: {str(e)}")
#                         import traceback
#                         traceback.print_exc()
            
#             # Also look for decoder layers that have self_attn
#             for n, m in n_m.items():
#                 if "DecoderLayer" in type(m).__name__ and hasattr(m, "self_attn"):
#                     attn_module = m.self_attn
#                     if not hasattr(attn_module, 'is_mla_wrapper'):  # Skip if already transformed
#                         try:
#                             # Create MLA module
#                             mla_module = llama2_to_mla_init(attn_module, {"config": config})
                            
#                             # Transform weights
#                             mla_module = transform_llama2_to_mla(attn_module, mla_module)
                            
#                             # Create wrapper with proper interface
#                             wrapped_module = MLAAttentionWrapper(mla_module)
                            
#                             # Replace directly in the decoder layer
#                             m.self_attn = wrapped_module
                            
#                             transformed_count += 1
#                             stats[type_name]["transformed_modules"].append(f"{n}.self_attn")
#                             print(f"Successfully transformed {n}.self_attn")
#                         except Exception as e:
#                             print(f"Error transforming {n}.self_attn: {str(e)}")
#                             import traceback
#                             traceback.print_exc()
            
#             # Skip the rest of the loop for "llama" type
#             continue
                            
#         # Rest of function for other module types...
#         # ...
        
#     print(f"Transformed {transformed_count} modules in total")
#     stats["total_transformed"] = transformed_count
#     return network, stats  # Return both network and stats

# def mla_by_name(network, pass_args):
#     is_huggingface_model = check_is_huggingface_model(network)

#     quantize_names = pass_args.keys()
#     n_m = {}
#     for n, m in network.named_modules():
#         n_m[n] = m
#     for n, m in n_m.items():
#         if n in quantize_names:
#             quan_config = pass_args[n]

#             quan_config = quan_config["config"]
#             postfix = quan_config.pop("name")

#             additional_module_args = (
#                 {"config": quan_config, "network_config": network.config}
#                 if is_huggingface_model
#                 else {"config": quan_config}
#             )

#             new_m = instantiate_module(
#                 m, postfix, attention_module_map, additional_module_args
#             )
#             network = replace_by_name(network, n, new_m)
#     return network


# def mla_by_regex_name(network, pass_args):
#     is_huggingface_model = check_is_huggingface_model(network)

#     patterns = list(pass_args.keys())
#     n_m = {}
#     for n, m in network.named_modules():
#         n_m[n] = m

#     for n, m in n_m.items():
#         matched_pattern = match_a_pattern(n, patterns)
#         if not matched_pattern:
#             continue

#         quan_config = pass_args[matched_pattern]["config"]
#         postfix = quan_config["name"]

#         additional_module_args = (
#             {"config": quan_config, "network_config": network.config}
#             if is_huggingface_model
#             else {"config": quan_config}
#         )

#         new_m = instantiate_module(
#             m, postfix, attention_module_map, additional_module_args
#         )
#         network = replace_by_name(network, n, new_m)

#     return network


# def attention_transform_pass(network, pass_args):
#     by = pass_args.pop("by")
#     stats = {}
#     match by:
#         case "type":
#             network, type_stats = mla_by_type(network, pass_args)
#             stats.update(type_stats)
#         case "name":
#             network = mla_by_name(network, pass_args)
#         case "regex_name":
#             network = mla_by_regex_name(network, pass_args)
#         case _:
#             raise ValueError(f'Unsupported quantize "by": {by}')
#     return network, stats



# import torch
# from typing import Optional
# import math
# from typing import Optional, Tuple, Union
# import logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# from chop.nn.attention.modules.mla import (
#     ModelArgs, 
#     MLA,
#     RMSNorm
# )
# from chop.nn.attention.modules.mgqa import (
#     MGQALayers,
# )
# from ...module_modify_helper import (
#     get_module_by_name, 
#     set_module_by_name,
# )
# from transformers.models.bert.modeling_bert import (
#     BertAttention,
#     BertSelfAttention, 
#     BertSdpaSelfAttention,
#     BertSelfOutput,
# )
# from transformers.models.gpt2.modeling_gpt2 import (
#     GPT2SdpaAttention,
#     GPT2Block,
# )
# from transformers.models.llama.modeling_llama import (
#     LlamaAttention,
#     LlamaDecoderLayer
# )

# def instantiate_attention_module(module, postfix, module_map, additional_module_args):
#     # sdpa_attn = module.self
#     # self_output = module.output
#     additional_module_args = additional_module_args["config"]
#     init_func = init_func_map[postfix]
    
#     attention_module = init_func(
#         module,
#         config=additional_module_args,
#     )

#     return attention_module

# def replace_attention_by_name(network, name, module, postfix):
    
#     original = get_module_by_name(network, name)

#     transform_func = transform_func_map[postfix]
#     wrapper_class = wrapper_map[postfix]

#     new = transform_func(original, module)
#     wapper = wrapper_class(new)

#     network = set_module_by_name(network, name, wapper)
#     return network


# def llama2_to_mla_init(
#     module,  
#     config: dict 
# ) -> MLA:
#     """
#     Initialize and return an MLA module based on dimensions
#     extracted from a Llama attention module.
    
#     Args:
#         module: Either a LlamaDecoderLayer or a LlamaSdpaAttention/LlamaAttention module
#         config (dict): Configuration dictionary
#     Returns:
#         MLA: A newly constructed MLA module with random initialization (not wrapped)
#     """
#     # Determine if we're dealing with a decoder layer or directly with an attention module
#     if hasattr(module, 'self_attn'):
#         # This is a LlamaDecoderLayer
#         llama_attention = module.self_attn
#     else:
#         # This is already an attention module
#         llama_attention = module

#     # Extract parameters from the attention module
#     if hasattr(llama_attention, 'hidden_size'):
#         hidden_size = llama_attention.hidden_size
#     elif hasattr(llama_attention, 'embed_dim'):
#         hidden_size = llama_attention.embed_dim
#     else:
#         hidden_size = llama_attention.q_proj.weight.shape[1]
        
#     if hasattr(llama_attention, 'num_heads'):
#         n_heads = llama_attention.num_heads
#     elif hasattr(llama_attention, 'num_attention_heads'):
#         n_heads = llama_attention.num_attention_heads
#     else:
#         head_dim = getattr(llama_attention, 'head_dim', 64)
#         n_heads = hidden_size // head_dim
        
#     head_dim = hidden_size // n_heads

#     print(f"Extracted dimensions: hidden_size={hidden_size}, n_heads={n_heads}, head_dim={head_dim}")

#     # Optional user config
#     user_config = config.get("config", {})

#     # Create ModelArgs for MLA
#     model_args = ModelArgs(
#         dim=hidden_size,
#         n_heads=n_heads,
#         qk_nope_head_dim=0,
#         qk_rope_head_dim=head_dim,
#         v_head_dim=head_dim,
#         max_seq_len=user_config.get("max_seq_len", 4096),
#         max_batch_size=user_config.get("max_batch_size", 4),
#         kv_lora_rank=min(hidden_size, 384)
#     )

#     # Construct MLA with those arguments
#     mla_module = MLA(model_args)
    
#     # Store model_args with the module for later use
#     mla_module.model_args = model_args

#     print(f"Created MLA with dimensions:")
#     print(f"  wq.weight: {mla_module.wq.weight.shape}")
#     print(f"  wkv_a.weight: {mla_module.wkv_a.weight.shape}")
#     print(f"  wkv_b.weight: {mla_module.wkv_b.weight.shape}")
#     print(f"  wo.weight: {mla_module.wo.weight.shape}")

#     # Return the unwrapped MLA module
#     return mla_module


# def transform_llama2_to_mla(
#     module,
#     mla_attn: MLA,
# ):
#     """
#     Transform weights from a Llama attention module to an MLA module.
    
#     Args:
#         module: Llama attention module
#         mla_attn (MLA): Target MLA module to be transformed
        
#     Returns:
#         MLA: Transformed MLA module (not wrapped)
#     """
#     # Determine source module
#     if hasattr(module, 'self_attn'):
#         llama_attention = module.self_attn
#     else:
#         llama_attention = module
    
#     # Extract weights
#     q_proj_weight = llama_attention.q_proj.weight
#     k_proj_weight = llama_attention.k_proj.weight
#     v_proj_weight = llama_attention.v_proj.weight
#     o_proj_weight = llama_attention.o_proj.weight
    
#     # Get target dtype
#     target_dtype = mla_attn.wq.weight.dtype
    
#     # Copy query weights
#     with torch.no_grad():
#         mla_attn.wq.weight.copy_(q_proj_weight.to(target_dtype))
    
#     print(f"MLA wkv_b.weight shape: {mla_attn.wkv_b.weight.shape}")
#     print(f"MLA wkv_a.weight shape: {mla_attn.wkv_a.weight.shape}")
    
#     # Concatenate k and v weights for low-rank decomposition
#     kv_weight = torch.cat([k_proj_weight, v_proj_weight], dim=0).to(torch.float32)
#     print(f"KV concatenated shape: {kv_weight.shape}")
    
#     # Get target dimensions
#     b_rows, b_cols = mla_attn.wkv_b.weight.shape
#     a_rows, a_cols = mla_attn.wkv_a.weight.shape
    
#     # Use proper rank
#     rank = min(b_cols, min(kv_weight.shape))
#     print(f"Using rank: {rank}")
    
#     # Compute SVD for low-rank approximation
#     try:
#         U, S, Vh = torch.linalg.svd(kv_weight, full_matrices=False)
#         print(f"SVD successful: U shape: {U.shape}, S shape: {S.shape}, Vh shape: {Vh.shape}")
        
#         # Truncate to rank
#         U_trunc = U[:, :rank]
#         S_trunc = torch.sqrt(S[:rank])
#         Vh_trunc = Vh[:rank, :]
        
#         # Create scaled A and B matrices
#         A = (U_trunc @ torch.diag(S_trunc)).to(torch.float32)
#         B = (torch.diag(S_trunc) @ Vh_trunc).to(torch.float32)
        
#         print(f"Created A shape: {A.shape}, B shape: {B.shape}")
        
#         # Create properly sized matrices
#         A_resized = torch.zeros((b_rows, b_cols), dtype=torch.float32, device=A.device)
#         B_resized = torch.zeros((a_rows, a_cols), dtype=torch.float32, device=B.device)
        
#         # Fill with values from A and B
#         repeat_rows_a = (b_rows + A.shape[0] - 1) // A.shape[0]
#         A_repeated = A.repeat(repeat_rows_a, 1)
#         A_resized[:, :b_cols] = A_repeated[:b_rows, :b_cols]
        
#         repeat_rows_b = (a_rows + B.shape[0] - 1) // B.shape[0]
#         B_repeated = B.repeat(repeat_rows_b, 1)
#         B_resized[:, :a_cols] = B_repeated[:a_rows, :a_cols]
        
#         print(f"Resized A shape: {A_resized.shape}, Resized B shape: {B_resized.shape}")
        
#         # Copy the factorized weights
#         with torch.no_grad():
#             mla_attn.wkv_b.weight.copy_(A_resized.to(target_dtype))
#             mla_attn.wkv_a.weight.copy_(B_resized.to(target_dtype))
    
#     except Exception as e:
#         print(f"SVD failed: {e}. Falling back to random initialization.")
#         with torch.no_grad():
#             torch.nn.init.normal_(mla_attn.wkv_b.weight, std=0.02)
#             torch.nn.init.normal_(mla_attn.wkv_a.weight, std=0.02)
    
#     # Adjust kv_norm if it exists
#     if hasattr(mla_attn, "kv_norm") and mla_attn.kv_norm is not None:
#         with torch.no_grad():
#             kv_norm_fill_value = 0.9 + 0.1 * torch.rand_like(mla_attn.kv_norm.weight)
#             mla_attn.kv_norm.weight.data.copy_(kv_norm_fill_value)
    
#     # Copy output projection weights
#     with torch.no_grad():
#         mla_attn.wo.weight.copy_(o_proj_weight.to(target_dtype))
    
#     # Return the transformed MLA (unwrapped)
#     return mla_attn

# class MLAWrapper(torch.nn.Module):
#     """
#     Wrapper for MLA to match LlamaAttention interface.
#     """
#     def __init__(self, mla_module):
#         super().__init__()
#         self.mla = mla_module
        
#         # Get dimensions from the mla module
#         self.hidden_size = mla_module.dim
#         self.num_heads = mla_module.n_heads
        
#         # Precompute frequency table once for efficiency
#         self.freqs_cis = precompute_freqs_cis(mla_module.model_args)
        
#         # Position counter for incremental decoding
#         self.register_buffer('position_counter', torch.zeros(1, dtype=torch.int), persistent=False)
        
#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         position_ids=None,
#         past_key_value=None,
#         output_attentions=False,
#         use_cache=False,
#         **kwargs
#     ):
#         """
#         Adapter between LlamaAttention interface and MLA interface.
#         """
#         batch_size, seq_len = hidden_states.size()[:2]
        
#         # Get target dtype from MLA parameters
#         param = next(self.mla.parameters(), None)
#         target_dtype = param.dtype if param is not None else torch.bfloat16
#         device = hidden_states.device
        
#         # Convert inputs to the target dtype
#         hidden_states = hidden_states.to(target_dtype)
        
#         # Convert attention mask format if needed
#         mla_mask = None
#         if attention_mask is not None:
#             if attention_mask.dim() == 4:
#                 mla_mask = attention_mask.squeeze(1)
#             else:
#                 mla_mask = attention_mask
            
#             # Convert mask to same dtype if it has non-boolean values
#             if mla_mask.dtype != torch.bool:
#                 mla_mask = mla_mask.to(target_dtype)
        
#         # Get start position for incremental decoding
#         start_pos = 0
#         if past_key_value is not None:
#             if isinstance(past_key_value, tuple) and len(past_key_value) == 2:
#                 start_pos = past_key_value[0].size(2)  # [bsz, num_heads, seq_len, head_dim]
#             elif hasattr(self, 'position_counter'):
#                 start_pos = self.position_counter.item()
#                 self.position_counter += seq_len
        
#         # Get appropriate freqs_cis slice
#         freqs_cis = self.freqs_cis
#         if position_ids is not None:
#             freqs_cis = torch.index_select(self.freqs_cis, 0, position_ids.view(-1))
#         else:
#             freqs_cis = self.freqs_cis[start_pos:start_pos+seq_len]
        
#         # Ensure freqs_cis is on the right device
#         freqs_cis = freqs_cis.to(device=device)
        
#         # Call MLA forward
#         output = self.mla(
#             x=hidden_states,
#             start_pos=start_pos,
#             freqs_cis=freqs_cis,
#             mask=mla_mask
#         )
        
#         # Convert output to match original dtype if needed
#         orig_dtype = kwargs.get('input_dtype', hidden_states.dtype)
#         if output.dtype != orig_dtype:
#             output = output.to(orig_dtype)
        
#         # Prepare outputs in Llama format
#         attn_weights = None
        
#         present_key_value = None
#         if use_cache:
#             head_dim = self.hidden_size // self.num_heads
#             dummy_key = torch.zeros(
#                 (batch_size, self.num_heads, seq_len, head_dim),
#                 device=device, dtype=orig_dtype
#             )
#             dummy_value = torch.zeros_like(dummy_key)
#             present_key_value = (dummy_key, dummy_value)
        
#         return output, attn_weights, present_key_value

# # In attention_transform_helper.py
# class MLAAttentionWrapper(torch.nn.Module):
#     """
#     Wrapper for MLA to match LlamaAttention interface.
#     Naming includes 'MLA' and 'Attention' to be easily detectable.
#     """
#     def __init__(self, mla_module):
#         super().__init__()
#         self.mla = mla_module
#         self.is_mla_wrapper = True  # Attribute flag for detection
        
#         # Get dimensions from the mla module
#         self.hidden_size = mla_module.dim
#         self.num_heads = mla_module.n_heads
        
#         # Register buffers - FIXED: only register freqs_cis once
#         self.register_buffer('position_counter', torch.zeros(1, dtype=torch.int), persistent=False)
#         self.register_buffer(
#             'freqs_cis',
#             precompute_freqs_cis(mla_module.model_args),
#             persistent=False
#         )
        
#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         position_ids=None,
#         past_key_value=None,
#         output_attentions=False,
#         use_cache=False,
#         **kwargs
#     ):
#         """
#         Adapter between LlamaAttention interface and MLA interface.
#         """
#         batch_size, seq_len = hidden_states.size()[:2]
        
#         # Get target dtype from MLA parameters
#         param = next(self.mla.parameters(), None)
#         target_dtype = param.dtype if param is not None else torch.bfloat16
#         device = hidden_states.device
        
#         # Convert inputs to the target dtype
#         hidden_states = hidden_states.to(target_dtype)
        
#         # Convert attention mask format if needed
#         mla_mask = None
#         if attention_mask is not None:
#             if attention_mask.dim() == 4:
#                 mla_mask = attention_mask.squeeze(1)
#             else:
#                 mla_mask = attention_mask
            
#             # Convert mask to same dtype if it has non-boolean values
#             if mla_mask.dtype != torch.bool:
#                 mla_mask = mla_mask.to(target_dtype)
        
#         # Get start position for incremental decoding
#         start_pos = 0
#         if past_key_value is not None:
#             if isinstance(past_key_value, tuple) and len(past_key_value) == 2:
#                 start_pos = past_key_value[0].size(2)  # [bsz, num_heads, seq_len, head_dim]
#             elif hasattr(self, 'position_counter'):
#                 start_pos = self.position_counter.item()
#                 self.position_counter += seq_len
        
#         # Get appropriate freqs_cis slice
#         freqs_cis = self.freqs_cis
#         if position_ids is not None:
#             # FIXED: Handle batched position_ids correctly
#             if position_ids.dim() > 1:
#                 # For MLA.apply_rotary_emb to work, freqs_cis must be [seq_len, head_dim/2]
#                 # When position_ids has batch dimension, use only the first batch's positions
#                 position_ids_flat = position_ids[0]
#                 freqs_cis = torch.index_select(self.freqs_cis, 0, position_ids_flat)
#             else:
#                 # Non-batched position_ids (1D tensor)
#                 freqs_cis = torch.index_select(self.freqs_cis, 0, position_ids)
#         else:
#             # No position_ids, use sequential positions
#             freqs_cis = self.freqs_cis[start_pos:start_pos+seq_len]
        
#         # Ensure freqs_cis is on the right device
#         freqs_cis = freqs_cis.to(device=device)
        
#         # Call MLA forward
#         try:
#             output = self.mla(
#                 x=hidden_states,
#                 start_pos=start_pos,
#                 freqs_cis=freqs_cis,
#                 mask=mla_mask
#             )
#         except Exception as e:
#             # Add debugging information if the forward call fails
#             import logging
#             logger = logging.getLogger(__name__)
#             logger.error("--- Error during self.mla forward call ---", exc_info=True)
#             logger.error(f"  hidden_states shape: {hidden_states.shape}")
#             logger.error(f"  start_pos: {start_pos}")
#             logger.error(f"  freqs_slice shape: {freqs_cis.shape if freqs_cis is not None else 'None'}")
#             logger.error(f"  mla_mask shape: {mla_mask.shape if mla_mask is not None else 'None'}")
#             raise e
        
#         # Convert output to match original dtype if needed
#         orig_dtype = kwargs.get('input_dtype', hidden_states.dtype)
#         if output.dtype != orig_dtype:
#             output = output.to(orig_dtype)
        
#         # Prepare outputs in Llama format
#         attn_weights = None
        
#         present_key_value = None
#         if use_cache:
#             head_dim = self.hidden_size // self.num_heads
#             dummy_key = torch.zeros(
#                 (batch_size, self.num_heads, seq_len, head_dim),
#                 device=device, dtype=orig_dtype
#             )
#             dummy_value = torch.zeros_like(dummy_key)
#             present_key_value = (dummy_key, dummy_value)
        
#         return output, attn_weights, present_key_value

# def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
#     """
#     Precomputes frequency-based complex exponential values for rotary positional embeddings.
#     """
#     dim = args.qk_rope_head_dim
#     seqlen = args.max_seq_len
#     beta_fast = args.beta_fast
#     beta_slow = args.beta_slow
#     base = args.rope_theta
#     factor = args.rope_factor

#     def find_correction_dim(num_rotations, dim, base, max_seq_len):
#         return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

#     def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
#         low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
#         high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
#         return max(low, 0), min(high, dim-1)

#     def linear_ramp_factor(min, max, dim):
#         if min == max:
#             max += 0.001
#         linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
#         ramp_func = torch.clamp(linear_func, 0, 1)
#         return ramp_func

#     # Create base frequencies (using float32 for precision)
#     freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    
#     # Apply YaRN scaling if needed
#     if seqlen > args.original_seq_len:
#         low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
#         smooth = 1 - linear_ramp_factor(low, high, dim // 2)
#         freqs = freqs / factor * (1 - smooth) + freqs * smooth

#     t = torch.arange(seqlen, dtype=torch.float32)
#     freqs = torch.outer(t, freqs)
    
#     # Convert to complex exponentials (stays in float32)
#     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
#     return freqs_cis


# init_func_map = {
#     "mla": llama2_to_mla_init,
#     "mgqa": gpt2sdpa_to_mgqa_init
# }

# transform_func_map = {
#     "mla": transform_llama2_to_mla,
#     "mgqa": transform_gpt2sdpa_to_mgqa,
# }

# wrapper_map = {
#     "mla": MLAAttentionWrapper,
#     "mgqa": MGQAWrapper,
# }