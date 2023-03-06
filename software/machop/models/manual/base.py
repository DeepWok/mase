import torch.nn as nn
import toml


class ManualBase(nn.Module):
    def __init__(self, config=None):
        super(ManualBase, self).__init__()
        if not config.endswith(".toml"):
            raise ValueError("Config file must be a toml file")
        self.config = toml.load(config)
        print(f"{self.config} loaded!")
