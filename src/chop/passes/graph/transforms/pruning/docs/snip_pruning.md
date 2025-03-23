# SNIP Pruning

## Overview

SNIP (Single-shot Network Pruning) is a pruning technique that identifies important connections in a neural network before training. Unlike traditional pruning methods that require a fully trained model, SNIP evaluates the importance of weights at initialization time, making it more efficient.

The key idea behind SNIP is to measure the sensitivity of the loss function with respect to each weight by calculating gradients through a binary mask applied to the weights. Weights with higher sensitivity (i.e., larger gradients) are considered more important.

## Reference

Lee, N., Ajanthan, T., & Torr, P. H. (2018). SNIP: Single-shot network pruning based on connection sensitivity. International Conference on Learning Representations (ICLR 2019). [https://arxiv.org/abs/1810.02340](https://arxiv.org/abs/1810.02340)

## How SNIP Works

1. **Connection Sensitivity**: SNIP defines the "connection sensitivity" as the magnitude of the gradient of the loss with respect to a binary mask multiplied by the corresponding weight.

2. **Single Forward-Backward Pass**: The method requires only a single mini-batch of data to compute these sensitivities with a forward and backward pass.

3. **One-shot Pruning**: Once the sensitivities are calculated, weights with the lowest sensitivities are pruned in a single step.

4. **Train the Pruned Network**: The pruned network is then trained normally from scratch.

## Implementation Details

In our implementation, SNIP pruning involves:

1. **SNIPTrackingCallback**: A callback that uses monkey-patching to calculate SNIP saliency scores for network weights. It adds temporary mask parameters to weights, performs a forward-backward pass, and captures gradients.

2. **SNIPCallback**: A class that implements the HuggingFace TrainerCallback interface, allowing SNIP sensitivity computation to be integrated with the training workflow.

3. **Pruning Functions**: 
   - `snip`: Performs local pruning within each layer based on sensitivity scores
   - `global_weight_snip`: Performs global pruning across all layers, with memory-efficient fallback for large models

## Usage

### Basic Usage

To use SNIP pruning in your code:

```python
import torch
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.transforms.pruning.snip_helper import SNIPTrackingCallback
from chop.passes.graph.transforms.pruning.prune import prune_transform_pass

# 1. Prepare your model and a batch of data
model = your_model
dummy_data = torch.randn(batch_size, *input_shape)

# 2. Create a MaseGraph from your model
mg = MaseGraph(model)
mg, _ = passes.init_metadata_analysis_pass(mg)
mg, _ = passes.add_common_metadata_analysis_pass(
    mg,
    pass_args={
        "dummy_in": dummy_data,
        "add_value": True,
        "force_device_meta": False,
    }
)

# 3. Compute SNIP saliency scores
callback = SNIPTrackingCallback()
callback(mg.model, dummy_data)

# 4. Apply pruning with SNIP method
pruning_config = {
    "weight": {
        "method": "snip",
        "granularity": "elementwise",
        "scope": "local",  # or "global" for global pruning
        "sparsity": 0.5     # Prune 50% of weights
    },
    "activation": {
        "method": "random",
        "granularity": "elementwise", 
        "scope": "local",
        "sparsity": 0.0     # No activation pruning
    }
}

mg, _ = prune_transform_pass(mg, pass_args=pruning_config)

# 5. Train the pruned model from scratch
# ...
```

### Using with HuggingFace Trainer

For integration with HuggingFace's Trainer:

```python
from transformers import Trainer, TrainingArguments
from chop.passes.graph.transforms.pruning.snip_helper import SNIPCallback

# Setup your model, tokenizer, dataset, etc.
# ...

# Create a representative batch for SNIP computation
dummy_batch = data_collator([dataset[i] for i in range(batch_size)])

# Initialize the Trainer with SNIPCallback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # Other trainer parameters...
)

# Add the SNIP callback
trainer.add_callback(SNIPCallback(representative_batch=dummy_batch))

# Train (the first step will compute SNIP scores)
trainer.train()

# Apply pruning with the scores
# ...
```

### Configuration Options

- **method**: Use "snip" for SNIP pruning.
- **granularity**: Currently only "elementwise" is supported for SNIP.
- **scope**:
  - "local": Apply pruning independently to each layer
  - "global": Apply pruning globally across all layers
- **sparsity**: The fraction of weights to prune (between 0 and 1).

## SNIP Workflow

1. **Sensitivity Score Calculation**
   - Replace each weight W with W⊙M (element-wise multiply with a mask)
   - Initialize mask M to all ones and make it learnable
   - Perform forward pass with a representative batch
   - Calculate loss and backpropagate
   - Store absolute gradient values |∂L/∂M| as sensitivity scores

2. **Pruning Based on Scores**
   - For local pruning: Calculate threshold for each layer separately
   - For global pruning: Calculate a single threshold across all layers
   - Create binary masks (0 or 1) based on thresholds
   - Apply masks to weights using PyTorch parametrization

3. **Train the Pruned Network**
   - Train the network from scratch with pruned weights

## Benefits of SNIP

- **Efficiency**: Pruning at initialization avoids the computational cost of training before pruning.
- **Performance**: SNIP typically maintains better performance compared to random pruning.
- **One-shot**: No iterative pruning process is needed.
- **Flexibility**: Can be used for both local (layer-wise) and global pruning.

## Limitations

- Requires a representative batch of data to compute saliency scores.
- May not be optimal for very deep networks or certain architectures.
- Does not consider the dynamics of training or the interaction between pruned connections.

## See Also

Check out the example in `examples/snip_example.py` for a complete demonstration of SNIP pruning on a speech recognition model. The example shows both the direct calculation approach and integration with the HuggingFace Trainer. 