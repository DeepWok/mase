#!/usr/bin/env python3
# Test script for transforming Llama model attention to MLA
import logging
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from chop.passes.module.transforms import attention_transform_pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_llama_mla_transform():
    """
    Test the transformation of Llama attention to MLA using only forward pass.
    Uses a TinyLlama model for testing.
    """
    # Use TinyLlama model from Hugging Face
    checkpoint = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small model for testing
    
    logger.info(f"Loading model from {checkpoint}")
    
    # Load the model and tokenizer
    try:
        # Load model from Hugging Face
        logger.info("Loading the model...")
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        
        # Load tokenizer
        logger.info("Loading the tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    # Test the model on a simple input with just a forward pass (no generation)
    logger.info("Running forward pass test on original model")
    input_text = "Explain the concept of"
    
    try:
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Set attention mask explicitly to avoid warnings
        if 'attention_mask' not in inputs:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            
        with torch.no_grad():
            outputs = model(**inputs)
        logger.info(f"Original model forward pass successful, output shape: {outputs.logits.shape}")
    except Exception as e:
        logger.error(f"Error running forward pass with original model: {e}")
        import traceback
        traceback.print_exc()
        return model, None

    # --------------------------------------------------
    # Transform the model's attention to MLA
    # --------------------------------------------------
    logger.info("Transforming model to use MLA attention")
    
    # Try to get the model architecture type to ensure we're transforming the right type
    model_type = model.config.model_type if hasattr(model.config, 'model_type') else "unknown"
    logger.info(f"Model type: {model_type}")
    
    # Determine transform key based on model type
    transform_key = "llama" if model_type.lower() == "llama" else "default"
    logger.info(f"Using transform key: {transform_key}")
    
    # Inspect the model structure before transformation
    logger.info("Inspecting model structure before transformation:")
    attention_modules = []
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if 'Attention' in module_type:
            attention_modules.append((name, module_type))
            logger.info(f"Found attention module: {name} of type {module_type}")
    
    if not attention_modules:
        logger.error("No Attention modules found in the model. Transformation will likely fail.")
    else:
        logger.info(f"Found {len(attention_modules)} attention modules to transform")
    
    # Check the structure of the first attention module as an example
    if attention_modules:
        first_module_name, _ = attention_modules[0]
        first_module = model.get_submodule(first_module_name)
        logger.info(f"First module structure: {first_module}")
        logger.info(f"First module attributes: {dir(first_module)}")
    
    pass_args = {
        "by": "type",
        transform_key: {
            "config": {
                "name": "mla",
            }
        }
    }
    
    # Log what should be happening
    logger.info(f"Transformation args: {pass_args}")
    logger.info("About to call attention_transform_pass...")
    
    try:
        # Apply the transformation
        mla_model, stats = attention_transform_pass(model, pass_args)
        logger.info(f"Transformation stats: {stats}")
        
        # If stats is empty, something might be wrong
        if not stats:
            logger.warning("Transformation returned empty stats - might indicate no changes were made")
    except Exception as e:
        logger.error(f"Error during transformation: {e}")
        import traceback
        traceback.print_exc()
        return model, None

    # --------------------------------------------------
    # Test the transformed model with just a forward pass (no generation)
    # --------------------------------------------------
    logger.info("Testing MLA-transformed model with forward pass")
    try:
        with torch.no_grad():
            # Run a forward pass with use_cache=False to avoid KV cache issues
            outputs = mla_model(**inputs, use_cache=False)
        logger.info(f"MLA model forward pass successful, output shape: {outputs.logits.shape}")
        
        # Log success
        logger.info("MLA transformation successful for forward pass!")
        logger.info("Note: Generation may not work due to KV cache handling in SimpleLlamaWrapper")
    except Exception as e:
        logger.error(f"Error running forward pass with MLA model: {e}")
        import traceback
        traceback.print_exc()

    # Check if any Attention modules remain after transformation
    logger.info("Comparing attention modules before and after transformation:")
    
    count_before = 0
    count_after = 0
    original_modules = set()
    transformed_modules = set()
    
    # Get detailed info about original modules
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if 'Attention' in module_type:
            count_before += 1
            original_modules.add(name)
            logger.info(f"Original module: {name} of type {module_type}")
    
    # Get detailed info about transformed modules
    for name, module in mla_model.named_modules():
        module_type = type(module).__name__
        if 'Attention' in module_type and 'MLA' not in module_type and 'Simple' not in module_type:
            count_after += 1
            transformed_modules.add(name)
            logger.info(f"Still present after transform: {name} of type {module_type}")
        elif 'SimpleLlamaWrapper' in module_type or 'MLA' in module_type:
            logger.info(f"New module after transform: {name} of type {module_type}")
    
    logger.info(f"Attention modules in original model: {count_before}")
    logger.info(f"Original Attention modules remaining in MLA model: {count_after}")
    
    # Check which modules were not transformed
    if original_modules == transformed_modules:
        print(original_modules, transformed_modules)
        logger.warning("No modules were transformed - same modules present before and after")
    else:
        transformed = original_modules - transformed_modules
        logger.info(f"Transformed modules: {transformed}")
    
    if count_after < count_before:
        logger.info("Transformation appears to have replaced some attention modules")
    else:
        logger.warning("No attention modules seem to have been replaced")
    
    # Examine the init_func_map to see what's registered
    try:
        from chop.passes.module.transforms.attention.attention_transform_helper import init_func_map, transform_func_map
        logger.info(f"init_func_map contains: {init_func_map.keys()}")
        logger.info(f"transform_func_map contains: {transform_func_map.keys()}")
        
        if 'mla' in init_func_map:
            logger.info(f"mla init function: {init_func_map['mla']}")
        else:
            logger.error("mla not found in init_func_map!")
    except Exception as e:
        logger.error(f"Error inspecting transform maps: {e}")

    print("\nDEBUG - Checking for MLAAttentionWrapper modules:")
    found_wrappers = 0
    mla_modules = []
    for name, module in mla_model.named_modules():
        module_type = type(module).__name__
        if 'MLAAttention' in module_type or hasattr(module, 'is_mla_wrapper'):
            found_wrappers += 1
            mla_modules.append(name)
            print(f"Found MLA wrapper: {name} of type {module_type}")
    print(f"Total MLA wrappers found: {found_wrappers}")

    return model, mla_model

if __name__ == "__main__":
    orig_model, mla_model = test_llama_mla_transform()
    if mla_model is not None:
        has_mla = any(
            isinstance(m, torch.nn.Module) and 
            ('MLA' in type(m).__name__ or hasattr(m, 'is_mla_wrapper'))
            for _, m in mla_model.named_modules()
        )
        if has_mla:
            logger.info("Test completed successfully with MLA modules found!")
        else:
            logger.error("Test completed but no MLA modules were found in the transformed model.")
    else:
        logger.error("Test failed - MLA model could not be created or tested.")