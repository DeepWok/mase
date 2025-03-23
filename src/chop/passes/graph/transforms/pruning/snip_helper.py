import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
import types
from transformers import TrainerCallback


class SNIPTrackingCallback:
    """
    SNIP Callback for calculating saliency scores using monkey-patching.
    
    For each prunable module (Conv2d/Linear) in the model:
      1. Add a learnable parameter 'weight_mask' (initialized to ones) if not present.
      2. Override the forward method so that it computes using (weight.detach() * weight_mask).
         This ensures that no gradients flow into the weight.
      3. Run one forward-backward pass on one mini-batch and capture the gradient on weight_mask.
      4. Store that gradient in module.metadata["weight"]["stats"]["snip_scores"].
      5. Restore the original forward method.
      
    Reference: https://arxiv.org/abs/1810.02340
    """
    
    def __init__(self, training_step_fn: Optional[Callable] = None):
        """
        Initialize the SNIP tracking callback.
        
        Args:
            training_step_fn: Optional function to use for the forward-backward pass.
                             If None, a default cross-entropy or MSE loss will be used.
        """
        self.training_step_fn = training_step_fn
        
    def __call__(self, graph):
        """
        Apply SNIP saliency score computation to the graph's model.
        
        Args:
            graph: The computational graph to analyze
            
        Returns:
            Updated graph with SNIP saliency scores
        """
        model = graph.model
        
        # Save original forward methods for later restoration
        original_forwards = {}
        
        # Collect prunable modules
        prunable_modules = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight"):
                prunable_modules.append((name, module))
                
        # No prunable modules found
        if not prunable_modules:
            print("No prunable modules found in the model.")
            return graph
        
        device = next(model.parameters()).device
        
        # Enable gradient computation
        was_training = model.training
        model.train()
        
        # Monkey-patch prunable modules
        for name, module in prunable_modules:
            original_forwards[name] = module.forward  # Save original forward
            
            # Create weight_mask parameter if not present
            if not hasattr(module, "weight_mask"):
                module.weight_mask = nn.Parameter(torch.ones_like(module.weight))
                
            # Override forward method to use weight.detach() * weight_mask
            if isinstance(module, nn.Conv2d):
                def new_forward(self, x):
                    return F.conv2d(
                        x,
                        self.weight.detach() * self.weight_mask,
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )
                module.forward = types.MethodType(new_forward, module)
            elif isinstance(module, nn.Linear):
                def new_forward(self, x):
                    return F.linear(x, self.weight.detach() * self.weight_mask, self.bias)
                module.forward = types.MethodType(new_forward, module)
        
        # Get a batch from the input metadata if available
        dummy_input = None
        for node in graph.fx_graph.nodes:
            if node.op == "placeholder":
                if "value" in node.meta.get("mase", {}).get("parameters", {}).get("common", {}).get("args", {}):
                    input_value = node.meta["mase"]["parameters"]["common"]["args"]["value"]
                    if isinstance(input_value, torch.Tensor):
                        if dummy_input is None:
                            dummy_input = {node.name: input_value.to(device)}
                        else:
                            dummy_input[node.name] = input_value.to(device)
        
        # If no dummy input found, use a fallback
        if dummy_input is None:
            print("Warning: No dummy input found in graph metadata. SNIP scores may be inaccurate.")
            # Try to infer input shape from the first layer
            for _, module in prunable_modules:
                if isinstance(module, nn.Conv2d):
                    in_channels = module.in_channels
                    dummy_input = {"input": torch.randn(1, in_channels, 28, 28).to(device)}
                    break
                elif isinstance(module, nn.Linear):
                    in_features = module.in_features
                    dummy_input = {"input": torch.randn(1, in_features).to(device)}
                    break
        
        # Zero gradients
        model.zero_grad()
        
        # Forward pass
        try:
            if isinstance(dummy_input, dict):
                output = model(**dummy_input)
            else:
                output = model(dummy_input)
            
            # Compute loss and backward
            if self.training_step_fn is not None:
                # Use provided training step function
                loss = self.training_step_fn(model, dummy_input, output)
            else:
                # Default loss computation
                if isinstance(output, dict) and "loss" in output:
                    # If model returns a dict with loss
                    loss = output["loss"]
                elif isinstance(output, torch.Tensor):
                    # Default to MSE loss against zeros (arbitrary target)
                    if output.dim() > 1 and output.size(1) > 1:
                        # Probably logits, use cross-entropy
                        target = torch.zeros(output.size(0), dtype=torch.long, device=device)
                        loss = F.cross_entropy(output, target)
                    else:
                        # Use MSE loss
                        target = torch.zeros_like(output)
                        loss = F.mse_loss(output, target)
                else:
                    raise ValueError("Could not determine how to compute loss from model output")
                    
            # Backward pass
            loss.backward()
            
        except Exception as e:
            print(f"Error during SNIP forward-backward pass: {e}")
            # Restore original forwards
            for name, module in prunable_modules:
                if name in original_forwards:
                    module.forward = original_forwards[name]
                    
            # Restore training mode
            model.train(was_training)
            return graph
        
        # For each prunable module, store the gradient of weight_mask as snip_scores
        for name, module in prunable_modules:
            grad = module.weight_mask.grad
            if grad is not None:
                # Calculate SNIP score: |weight * grad|
                snip_scores = torch.abs(grad)
                
                # Store in module.metadata
                if not hasattr(module, "metadata"):
                    module.metadata = {}
                if "weight" not in module.metadata:
                    module.metadata["weight"] = {}
                if "stats" not in module.metadata["weight"]:
                    module.metadata["weight"]["stats"] = {}
                
                # Store the SNIP saliency scores
                module.metadata["weight"]["stats"]["snip_scores"] = snip_scores
                
                print(f"Module {name}: SNIP score norm = {snip_scores.norm().item()}")
            else:
                print(f"Module {name}: no SNIP score computed (grad is None)")
        
        # Restore original forward methods
        for name, module in prunable_modules:
            if name in original_forwards:
                module.forward = original_forwards[name]
        
        # Update FX graph metadata with SNIP scores
        for node in graph.fx_graph.nodes:
            if node.op == "call_module":
                module_name = node.target
                module = getattr(graph.model, module_name, None)
                
                if module is not None and hasattr(module, "metadata"):
                    module_metadata = module.metadata
                    if "weight" in module_metadata and "stats" in module_metadata["weight"]:
                        if "snip_scores" in module_metadata["weight"]["stats"]:
                            # Add the scores to the node's metadata
                            if "software" not in node.meta.get("mase", {}).get("parameters", {}):
                                node.meta.setdefault("mase", {}).setdefault("parameters", {})["software"] = {"args": {}}
                            if "args" not in node.meta["mase"]["parameters"]["software"]:
                                node.meta["mase"]["parameters"]["software"]["args"] = {}
                            if "weight" not in node.meta["mase"]["parameters"]["software"]["args"]:
                                node.meta["mase"]["parameters"]["software"]["args"]["weight"] = {"stat": {}}
                            if "stat" not in node.meta["mase"]["parameters"]["software"]["args"]["weight"]:
                                node.meta["mase"]["parameters"]["software"]["args"]["weight"]["stat"] = {}
                            
                            # Store the SNIP scores
                            node.meta["mase"]["parameters"]["software"]["args"]["weight"]["stat"]["snip_scores"] = \
                                module_metadata["weight"]["stats"]["snip_scores"]
        
        # Restore training mode
        model.train(was_training)
        
        return graph


class SNIPCallback(TrainerCallback):
    """Callback for applying SNIP sensitivity analysis during training.
    
    This callback can be used with a Hugging Face Trainer to compute and store
    SNIP sensitivity scores in the module metadata during the first training step.
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
        import types
        import torch.nn as nn
        import torch.nn.functional as F
        
        model = kwargs.get("model", None)
        if model is None:
            raise ValueError("No model provided to SNIP callback.")
        
        device = next(model.parameters()).device
        
        # Save the original forward methods of all prunable modules.
        original_forwards = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight"):
                original_forwards[name] = module.forward
                
                # Create weight_mask parameter if not present
                if not hasattr(module, "weight_mask"):
                    module.weight_mask = nn.Parameter(torch.ones_like(module.weight))
                
                # Override forward methods to use weight_mask
                if isinstance(module, nn.Conv2d):
                    def new_forward(self, x):
                        return F.conv2d(
                            x,
                            self.weight.detach() * self.weight_mask,
                            self.bias,
                            self.stride,
                            self.padding,
                            self.dilation,
                            self.groups,
                        )
                    module.forward = types.MethodType(new_forward, module)
                elif isinstance(module, nn.Linear):
                    def new_forward(self, x):
                        return F.linear(x, self.weight.detach() * self.weight_mask, self.bias)
                    module.forward = types.MethodType(new_forward, module)

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
        
        # Run one forward-backward pass
        model.zero_grad()
        output = model(**inputs)
        loss = output.get("loss", None)
        if loss is None:
            raise ValueError("Loss is None in SNIP callback; ensure your model returns a loss.")
        
        loss.backward()

        # For each prunable module, store the gradient of weight_mask in metadata.
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight_mask"):
                grad = module.weight_mask.grad
                if grad is not None:
                    if not hasattr(module, "metadata"):
                        module.metadata = {}
                    if "weight" not in module.metadata:
                        module.metadata["weight"] = {}
                    if "stats" not in module.metadata["weight"]:
                        module.metadata["weight"]["stats"] = {}
                    
                    # Store the SNIP scores (absolute gradient values)
                    module.metadata["weight"]["stats"]["snip_scores"] = grad.abs().detach().clone()
                    print(f"Module {name}: SNIP score norm = {grad.abs().norm().item()}")
                else:
                    print(f"Module {name}: no SNIP score computed (grad is None)")

        # Restore the original forward methods.
        for name, module in model.named_modules():
            if name in original_forwards:
                module.forward = original_forwards[name]
        
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