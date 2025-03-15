import torch
from torch import nn
from transformers import TrainerCallback

class MovementTrackingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.prev_params = {}
        self.movement_stats = {}
        model = kwargs["model"]
        for name, module in model.encoder.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight"):
                self.prev_params[name] = module.weight.detach().clone()
                self.movement_stats[name] = torch.zeros_like(module.weight)
                if not hasattr(module.weight, "metadata"):
                    module.metadata["weight"] = {}
                module.metadata["weight"]["stats"] = {"movement": self.movement_stats[name]}
        return control

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        for name, module in model.encoder.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight"):
                movement = (module.weight.detach() - self.prev_params[name]).abs()
                self.movement_stats[name] += movement
                self.prev_params[name].copy_(module.weight.detach())
                if not hasattr(module, "metadata"):
                    module.metadata = {}
                if "weight" not in module.metadata:
                    module.metadata["weight"] = {}
                module.metadata["weight"]["stats"] = {"movement": self.movement_stats[name]}

        return control

