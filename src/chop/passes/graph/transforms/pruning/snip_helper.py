import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
import types
from transformers import TrainerCallback


class SNIPCallback(TrainerCallback):
    """Callback for applying SNIP sensitivity analysis during training.
    
    This callback integrates with Hugging Face's Trainer API to compute and store
    SNIP sensitivity scores in the module metadata during the first training step.
    
    This is the recommended way to use SNIP pruning with Hugging Face Transformers models.
    
    Example:
        ```python
        # Create the callback with a representative batch
        snip_callback = SNIPCallback(representative_batch=batch)
        
        # Add to trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[snip_callback],
        )
        
        # Run training (SNIP will be computed at the beginning)
        trainer.train()
        ```
    """
    
    def __init__(self, representative_batch=None):
        """Initialize the SNIP callback.
        
        Args:
            representative_batch: Optional batch of data to use for SNIP computation.
                If not provided, will use the first batch from the training dataloader.
        """
        self.representative_batch = representative_batch
        self.snip_computed = False
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Compute SNIP scores at the beginning of training.
        
        This method is called by the Hugging Face Trainer at the start of training.
        It will compute SNIP scores for all prunable layers in the model.
        
        Args:
            args: Arguments passed to the Trainer.
            state: Current training state.
            control: Training control variables.
            **kwargs: Additional keyword arguments.
        
        Returns:
            control: Updated training control variables.
        """
        model = kwargs.get("model", None)
        if model is None:
            raise ValueError("No model provided to SNIP callback.")
        
        device = next(model.parameters()).device
        
        # Get a batch of data
        inputs = None
        
        # Option 1: Use representative batch passed during initialization
        if self.representative_batch is not None:
            print("Using provided representative batch for SNIP computation")
            inputs = self.representative_batch
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
        
        # Option 2: Get the first batch from the trainer's dataloader
        elif hasattr(kwargs.get("trainer", {}), "get_train_dataloader"):
            print("Using first batch from trainer's dataloader for SNIP computation")
            train_dataloader = kwargs["trainer"].get_train_dataloader()
            try:
                inputs = next(iter(train_dataloader))
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device)
            except StopIteration:
                raise ValueError("Empty training dataloader in SNIP callback")
        
        # Option 3: Use the train_dataloader provided in kwargs
        elif kwargs.get("train_dataloader") is not None:
            print("Using provided train_dataloader for SNIP computation")
            try:
                inputs = next(iter(kwargs["train_dataloader"]))
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device)
            except StopIteration:
                raise ValueError("Empty training dataloader in SNIP callback")
        
        # No data found
        if inputs is None:
            raise ValueError(
                "No data available for SNIP computation. Please provide either:"
                "\n1. A representative_batch when initializing SNIPCallback"
                "\n2. Ensure the trainer has a working get_train_dataloader method"
                "\n3. Pass train_dataloader in kwargs"
            )
        
        # Use the internal implementation to compute SNIP scores
        snip_impl = _SNIPTrackingImplementation()
        snip_impl.apply_to_model(model, inputs)
        
        # Mark SNIP as computed so we don't recompute
        self.snip_computed = True
        
        return control
    
    # Implement required TrainerCallback methods
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of an epoch."""
        return control
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of an epoch."""
        return control
        
    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of a training step."""
        return control
        
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of a training step."""
        return control
        
    def on_evaluate(self, args, state, control, **kwargs):
        """Called after evaluation."""
        return control
        
    def on_save(self, args, state, control, **kwargs):
        """Called when saving the model."""
        return control
        
    def on_log(self, args, state, control, **kwargs):
        """Called when logging metrics."""
        return control
        
    def on_prediction_step(self, args, state, control, **kwargs):
        """Called during evaluation on a prediction step."""
        return control
        
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        return control 