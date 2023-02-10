import torch
import logging
import toml
import pprint

from torch import nn
from .quantizers import ops_map
from collections import OrderedDict

pp = pprint.PrettyPrinter(depth=4)


class Modifier:

    modifiable_layers = ['linear', 'relu']

    def __init__(self, model=None, config=None, save_name=None):
        self.model = model

        #load config as toml
        if not config.endswith('.toml'):
            raise ValueError('Config file must be a toml file')
        config = toml.load(config)
        logging.info(f'Config loaded!')
        pp.pprint(config)
        self.config = config 

        self._pre_modify_check()
        logging.info(f"Model architecture")
        print(self.model)

        # The core function that makes the modification
        self.modify()

        logging.info(f"Model architecture, post modification")
        print(self.model)
        self._post_modify_check()
        self._print_out_diff()

        if save_name is not None:
            self.save(save_name)

    def modify(self):
        default_config = self.config.pop('default', None)
        if default_config is None:
            raise ValueError('Default config is not provided')

        for name in self.modifiable_layers:
            # use config setup if we have a config
            if name in self.config:
                replace_config = self.config[name]
                replace_fn = getattr(self, f'replace_{name}')
                replace_fn(self.model, replace_config)
            # otherwise all modifiable layers are changed based on default config
            else:
                replace_fn = getattr(self, f'replace_{name}')
                replace_fn(self.model, default_config)
        logging.info('Model modified')
    
    def _pre_modify_check(self):
        self._pre_modify_values = self.model.state_dict()
    
    def _post_modify_check(self):
        self._post_modify_values = OrderedDict()
        for name, module in self.model.named_modules():
            if hasattr(module, 'get_quantized_weight'):
                value = module.get_quantized_weight()
                self._post_modify_values[name] = value
    
    def _print_out_diff(self):
        # printout 10 numbers in each tensor
        logging.info("A pintout, each tensor only outputs 5 values")
        logging.info("Pre-modification values")
        for k, v in self._pre_modify_values.items():
            print(k)
            print(v.flatten()[:5])

        logging.info("Post-modification values")
        for k, v in self._post_modify_values.items():
            print(k)
            print(v.flatten()[:5])

    def replace_linear(self, model, config):
        replace_cls = ops_map['linear'][config['name']]
        target = nn.Linear
        # This shows how to use the replace function
        # First, we have to define a custom replacement_fn
        # We then call the replace function with the model, target layer, and replacement_fn
        def replacement_fn(child):
            # Check https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
            # for details about the class definition of child
            use_bias = child.bias is not None
            # class instantiation
            my_linear = replace_cls(
                in_features=child.in_features, 
                out_features=child.out_features, 
                bias=use_bias, 
                config=config)
            # grab pretrained weights
            # WARNING: need to test on the gpu land!
            my_linear.weight = child.weight.cpu()
            my_linear.bias = child.bias.cpu()
            return my_linear
        self.replace(model, target, replacement_fn)

    def replace_relu(self, model, config):
        replace_cls = ops_map['relu'][config['name']]
        target = nn.ReLU
        def replacement_fn(child):
            return replace_cls(
                inplace=child.inplace, 
                config=config)
        self.replace(model, target, replacement_fn)

    # A generic replacement function that works for any layer
    # This function traverses the modules in a model recursively
    # And replaces the target layer with the replacement layer
    def replace(self, model, target, replacement_fn):
        for child_name, child in model.named_children():
            if isinstance(child, target):
                setattr(model, child_name, replacement_fn(child))
            else:
                self.replace(child, target, replacement_fn)
    
    def save(self, name):
        torch.save(self.model.state_dict(), name)
        logging.info(f"Modified model saved as {name}.")