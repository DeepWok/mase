from copy import copy, deepcopy
import logging
import torch
from torch_tensorrt import torchtrt

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# pip install --no-cache-dir --index-url https://pypi.nvidia.com pytorch-quantization !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



from pytorch_quantization import quant_modules, calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn import TensorQuantizer
from pytorch_quantization.tensor_quant import QuantDescriptor
``
from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass
from ....utils import deepcopy_mase_graph


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]

def pytorch_quantize_by_type(graph, config: dict):        
    # Apply symbolic tracing to create the FX graph
    if 'default' in config:
        model = graph.model
        # Monkey-patch the model with quantization modules
        quant_modules.initialize(model, dtype=torch.quint8)

        # Freeze weights to avoid further updates during quantization calibration
        for param in model.parameters():
            param.requires_grad = False

        # Calibrate the model with representative input data
        # Replace this with your own calibration step
        dummy_input = config["train_generator"]
        model.eval()
        quant_modules.prepare(model, calib_data=[dummy_input])

        # Convert the model to a quantized state
        quant_modules.convert(model)

        # Save the quantized model
        torch.save(model.state_dict(), "quantized_model.pt")

        print("Successfully quantized the model using monkey-patching!")

        # fx_model = torch.fx.symbolic_trace(graph.model)

        # quant_modules.initialize()
        # TensorQuantizer.use_fb_fake_quant = True  # Use fake quantization for calibration

        # # Prepare the model by inserting quantization nodes
        # quant_desc_input = QuantDescriptor(calib_method='histogram')
        # quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
        # quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

        # # model = fx_model
        # # model.cuda()

        # # Calibrate the model (replace 'calibration_loader' with your data loader)
        # for inputs, _ in config["train_generator"]:
        #     quant_modules(inputs)

        # # Convert the model to a quantized version
        # quantized_model = quant_modules.convert_fx(quant_modules)

        # # Evaluate the model (replace 'evaluation_loader' with your data loader)
        # quantized_model.eval()
        # with torch.no_grad():
        #     for inputs, labels in config["val_generator"]:
        #         outputs = quantized_model(inputs)
        #         # Evaluate your model’s performance

        # # Convert the modified FX graph back to a PyTorch model
        # quantized_model = fx_model.to_pytorch_module()

        graph.model = quantized_model

    return graph

    



def pytorch_quantize_by_name(graph, config: dict):
    raise NotImplementedError()


def pytorch_quantize_by_regex_name(graph, config: dict):
    raise NotImplementedError()


def tensorrt_quantize_transform_pass(graph, pass_args=None):
    by = pass_args.pop("by")
    match by:
        case "type":
            graph = pytorch_quantize_by_type(graph, pass_args)
        case "name":
            graph = pytorch_quantize_by_name(graph, pass_args)
        case "regex_name":
            graph = pytorch_quantize_by_regex_name(graph, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}
