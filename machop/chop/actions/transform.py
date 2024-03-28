import os
from copy import deepcopy
from pathlib import Path
import logging

import torch
import pytorch_lightning as pl
import copy
import pdb
from chop.passes.graph import PASSES
from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    init_metadata_analysis_pass,
)
from chop.ir.graph.mase_graph import MaseGraph
from chop.passes.graph.interface import (
    load_mase_graph_interface_pass,
    save_mase_graph_interface_pass,
)
from chop.passes.graph.utils import deepcopy_mase_graph
from chop.tools.checkpoint_load import load_model
from chop.tools.config_load import load_config
from chop.tools.get_input import InputGenerator, get_cf_args, get_dummy_input
from chop.tools.utils import parse_accelerator, to_numpy_if_tensor

from chop.passes.graph.transforms import metadata_value_type_cast_transform_pass
from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target
import pprint
from chop.plt_wrapper import get_model_wrapper

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from chop.tools.config_load import load_config

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter(indent=4)


def pre_transform_load(model_short_name, mask, is_quantize, load_name: str, load_type: str, model: torch.nn.Module):
    if load_name is not None and load_type in ["pt", "pl"]:
        model = load_model(model_short_name, mask, is_quantize, load_name=load_name, load_type=load_type, model=model)
        '''
        load_model is added with parameters of 'mask' and 'is_quantize';
        1. becuase after pruning, there should be a group of new keys in state_dict(), which is "mask"
        e.g. state_dict['feature_layers.0.parametrizations.weight.0.mask'] = mask[0]
        2. becuase if quantized, the weight parameters will have parametrizations;
        e.g. change from 'feature_layers.0.weight' to 'feature_layers.0.parametrizations.weight.original'
        '''
    return model

# compute the model storage size
def model_storage_size(model, is_quantize, dict_weight_masks):
    if is_quantize:
        weight_bit_width = 8
        bias_bit_width = 8
    else:
        weight_bit_width = 32
        bias_bit_width = 32
    total_bits = 0 
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if dict_weight_masks != {}:
                if hasattr(module, 'weight'):
                    name = name.rsplit('.')
                    name = '_'.join(name)
                    mask = dict_weight_masks[name]
                    if mask is not None:
                        num_unpruned = torch.sum(mask).item()
                        bits = num_unpruned * weight_bit_width
            else:
                bits = module.weight.numel() * weight_bit_width
            total_bits += bits
        if hasattr(module, 'bias') and module.bias is not None:
            bits = module.bias.numel() * bias_bit_width
            total_bits += bits
    total_bytes = total_bits / 8
    return total_bytes


# only compute the flop of convolution layers 
def conv_flop(model, act_masks, dict_weight_masks):
    # FLOP: K * K * H_out * W_out * C_out * C_in * 2
    conv_flop_before_prune = 0 ; conv_flop_after_prune = 0
    assert len(act_masks) == len(dict_weight_masks.keys()), "do not have the same number of elements within activation masks and weight masks, please check"
    i=0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            H_out = act_masks[i].shape[-2] ; W_out = act_masks[i].shape[-1]
            name = name.rsplit('.') ; name = '_'.join(name)
            C_out = dict_weight_masks[name].shape[0] ; C_in = dict_weight_masks[name].shape[1] ; K = dict_weight_masks[name].shape[2]
            conv_flop_before_prune += (K * K * H_out * W_out) * C_in * C_out * 2
            # prune
            remain_percent = torch.sum(act_masks[i]>0) / (act_masks[i].shape[0] * act_masks[i].shape[1] * act_masks[i].shape[2] * act_masks[i].shape[3])
            conv_flop_after_prune += int((K * K * H_out * W_out) * (C_in * remain_percent) * C_out * 2)
            i += 1
    return conv_flop_before_prune, conv_flop_after_prune



def transform(
    model: torch.nn.Module,
    model_info,
    model_name,
    data_module,
    dataset_info,
    task,
    config,
    visualizer,
    prune_save_dir: str = None,
    quantize_save_dir: str=None,
    retrain_save_path: str=None,
    huffman_save_dir: str=None,
    load_name: str = None,
    load_type: str = None,
    accelerator: str = "auto",
):
    is_quantize = False
    accelerator = parse_accelerator(accelerator)  # cpu or gpu
    config = load_config(config) # config (basics & prune & quantize & retrain & huffman)

    load_name = config['passes']['retrain']['load_name']  
    load_type = config['passes']['retrain']['load_type']  # pt

    
    model_short_name = config['model'];print(model_short_name)
    dataset_short_name = config['dataset'];print(dataset_short_name)

    weight_mask = None
    if "vgg" in model_short_name:
        if "cifar" in dataset_short_name:
            model = pre_transform_load(model_short_name, weight_mask, is_quantize, load_name=load_name, load_type=load_type, model=model)  # load the VGG_pretrained model
            # the reason for 'weight_mask' and 'is_quantize' are given above
            model.to(accelerator)

    if "cf_args" not in config:
        cf_args = get_cf_args(model_info=model_info, task=task, model=model)
    else:
        cf_args = config["cf_args"]

    # graph generation
    graph = MaseGraph(model=model, cf_args=cf_args)
    # graph_metadata = Mase
    graph, _ = init_metadata_analysis_pass(graph, pass_args=None)

    # create or load metadata.parameters and mase_graph.model
    if load_name is not None and load_type == "mz":
        graph, _ = load_mase_graph_interface_pass(graph, pass_args=load_name)
    else:
        dummy_in = get_dummy_input(
            model_info=model_info,
            data_module=data_module,
            task=task,
            device=accelerator,
        )
        if len(graph.model.additional_inputs) > 0:
            dummy_in = dummy_in | graph.model.additional_inputs
        graph, _ = add_common_metadata_analysis_pass(
            graph, pass_args={"dummy_in": dummy_in}
        )
        graph, _ = add_software_metadata_analysis_pass(graph, pass_args=None)

    pass_config = config["passes"]
    huffman_pass_config = copy.deepcopy(pass_config)

    for pass_name, pass_config in pass_config.items():
        pass_name: str
        pass_config: dict
        match pass_name:
            case "quantize":
                graph, _ = metadata_value_type_cast_transform_pass(
                    graph, pass_args={"fn": to_numpy_if_tensor}
                )
                graph, _ = PASSES["quantize"](graph, pass_args=pass_config)
                is_quantize = True

                for n in graph.nodes:
                    if isinstance(get_node_actual_target(n), torch.nn.modules.Conv2d): 
                        if 'mase' in n.meta:
                            quantized_weight = get_node_actual_target(n).w_quantizer(get_node_actual_target(n).weight)
                            graph.model.state_dict()['.'.join(n.name.rsplit('_', 1)) + ".weight"].copy_(quantized_weight)
                            print(f"There is quantization at {n.name}, mase_op: {get_mase_op(n)}")

                # model size after quantization:
                model_size_after_quantize = model_storage_size(graph.model, is_quantize, dict_weight_masks)
                print("model size after quantization: ", model_size_after_quantize)

                # save the quantized model
                save_dir = quantize_save_dir ; save_dir = Path(save_dir) ; save_dir.mkdir(parents=True, exist_ok=True)           
                graph, _ = metadata_value_type_cast_transform_pass(graph, pass_args={"fn": to_numpy_if_tensor})
                graph, _ = save_mase_graph_interface_pass(graph, pass_args=save_dir) 
                logger.info(f"model is successfully quantized and saved to {save_dir}!")

            case "profile_statistics":
                input_generator = InputGenerator(
                    model_info=model_info,
                    data_module=data_module,
                    task=task,
                    which_dataloader="train",
                )
                pass_config["input_generator"] = input_generator
                graph, _ = PASSES[pass_name](graph, pass_args=pass_config)
            case "report_graph":
                pass_file_name = pass_config.get(
                    "file_name", save_dir / "report_graph.txt"
                )
                graph, _ = PASSES[pass_name](graph, file_name=pass_file_name)
            case "report_node_type":
                graph, _ = PASSES[pass_name](graph, pass_args=None)
            case "report_node_meta_param":
                # {"save_path": ..., "which": "all"|["common", "hardware", "software"]}
                pass_save_path = pass_config.get("save_path", save_dir / "report")
                pass_config["save_path"] = pass_save_path
                graph, _ = PASSES[pass_name](graph, pass_args=pass_config)
            case "report_node_shape":
                graph, _ = PASSES[pass_name](graph, pass_args=None)
            case "report_node_type":
                graph, _ = PASSES[pass_name](graph, pass_args=None)
            case "report_node_hardware_type":
                graph, _ = PASSES[pass_name](graph, pass_args=None)
            case "report_node_shape":
                graph, _ = PASSES[pass_name](graph, pass_args=None)
            case "report_node_type":
                graph, _ = PASSES[pass_name](graph, pass_args=None)
            case "load_mase_graph":
                pass_load_dir = pass_config["load_dir"]
                graph, _ = PASSES[pass_name](graph, pass_args=pass_load_dir)
            case "load_node_meta_param":
                pass_load_path = pass_config["load_path"]
                graph, _ = PASSES[pass_name](graph, pass_args=pass_load_path)
            case "save_mase_graph":
                pass_save_dir = pass_config.get(
                    "save_dir", save_dir / "saved_mase_graph"
                )
                graph, _ = PASSES[pass_name](graph, pass_args=pass_save_dir)
            case "save_node_meta_param":
                pass_save_path = pass_config.get(
                    "save_path", save_dir / "saved_node_meta_param"
                )
                graph, _ = PASSES[pass_name](graph, pass_args=pass_save_path)
            case "prune":
                input_generator = InputGenerator(  
                    model_info=model_info,
                    data_module=data_module,
                    task=task,
                    which_dataloader="val",
                )
                print("pass_config") ; print(pass_config)
                pass_config["model_name"] = model_name
                pass_config["input_generator"] = input_generator
                batch_size = config['passes']['retrain']['training']['batch_size']

                num_conv_param_before_prune = sum(p.numel() for m in model.modules() if isinstance(m, torch.nn.Conv2d) for p in m.parameters())

                dict_weight_masks={} # has the same value as weight_masks, but will be a form of dict
                model_size_before_prune = model_storage_size(graph.model, is_quantize, dict_weight_masks)

                
                # pruning process
                graph, _ = PASSES[pass_name](
                    graph,
                    batch_size,
                    pass_config,
                )  

                # calculate the pruning sparsity
                graph, sparsity_info, weight_masks, act_masks= PASSES["add_pruning_metadata"](
                # graph, sparsity_info, weight_masks, act_masks= PASSES["add_pruning_metadata"](
                    graph,
                    {"dummy_in": dummy_in, "add_value": False}
                )

                # weight pruning is of static process, where the weight mask for each layer remains during fine-tuning
                # activation pruning is of dynamic pruning, where the activation mask for each input batch )iteration) will be updated

                # if we want to force activation pruning to be static, we could run the following two lines to save activation masks
                #torch.save(act_masks, "act_masks.pth")
                #print("activation mask saved")
                
                pp.pprint(sparsity_info) 
                #del act_masks # reduce memory
                
                # calculate model size and FLOP
                num_conv_param_after_prune = 0
                for node in graph.fx_graph.nodes:
                      if node.op == "call_module":
                            if isinstance(graph.modules[node.target], torch.nn.Conv2d):
                                mask = graph.modules[node.target].parametrizations.weight[0].mask
                                dict_weight_masks[node.name] = mask
                                num_true = torch.sum(mask).item()
                                num_conv_param_after_prune += num_true
                model_size_after_prune = model_storage_size(graph.model, is_quantize, dict_weight_masks)

                # flop of Conv2d layers
                conv_flop_before_prune, conv_flop_after_prune = conv_flop(graph.model,act_masks,dict_weight_masks)
                
                # print 1)number of parameters  2)model size  3)flop of Conv2d  before and after pruning:
                print("-------------------------------------")
                print("number of Conv2d parameters before pruning: ", num_conv_param_before_prune)
                print("model size before pruning: ", model_size_before_prune)
                print("flop of Conv2d layers before pruning: ", conv_flop_before_prune)
                print("-------------------------------------")
                print("number of Conv2d parameters after pruning: ", num_conv_param_after_prune)
                print("model size after pruning: ", model_size_after_prune)
                print("flop of Conv2d layers after pruning: ", conv_flop_after_prune)
                print("-------------------------------------")
                print("reduced percentage of Conv2d parameters: ", 1-num_conv_param_after_prune/num_conv_param_before_prune)
                print("reduced percentage of model size: ", 1-model_size_after_prune/model_size_before_prune)
                print("reduced percentage of Conv2d flops: ", 1-conv_flop_after_prune/conv_flop_before_prune)
                print("-------------------------------------")
                
                # save the pruned model
                save_dir = prune_save_dir ; save_dir = Path(save_dir) ; save_dir.mkdir(parents=True, exist_ok=True)           
                graph, _ = metadata_value_type_cast_transform_pass(graph, pass_args={"fn": to_numpy_if_tensor})
                graph, _ = save_mase_graph_interface_pass(graph, pass_args=save_dir) 
                logger.info(f"model is successfully pruned and saved to {save_dir}!")
            
            case "retrain": # fine-tuning
                from pytorch_lightning.callbacks import Callback
                class HessianComputationCallback(Callback):
                    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                        loss = outputs['loss']
                        named_parameters = list(pl_module.named_parameters())
                        name, param = named_parameters[1]
                        if 'weight' in name:
                            hessian_diag = self.compute_hessian_diag(param, pl_module, loss)
                            print(f"[Batch {batch_idx}] Hessian Diagonal for {name}: max={hessian_diag.max().item()}, min={hessian_diag.min().item()}, mean={hessian_diag.mean().item()}")
                    @staticmethod
                    def compute_hessian_diag(param, model, loss):
                        model.eval()
                        loss.requires_grad_(True)
                        first_order_grads = torch.autograd.grad(loss, param, create_graph=True, allow_unused=True)

                        hessian_diag = []
                        for grad in first_order_grads:
                            if grad is not None:
                                grad_grad = torch.autograd.grad(grad, param, retain_graph=True)[0]
                                hessian_diag.append(grad_grad)

                        hessian_diag = torch.stack(hessian_diag).view_as(param)
                        return hessian_diag

                plt_trainer_args={}
                if retrain_save_path is not None:
                    # if retrain_save_path is None, the model will not be saved
                    if not os.path.isdir(retrain_save_path):
                        os.makedirs(retrain_save_path)
                    checkpoint_callback = ModelCheckpoint(
                        save_top_k=1,
                        monitor="val_loss_epoch",
                        mode="min",
                        filename="best",
                        dirpath=retrain_save_path,
                        save_last=True,
                    )
                    hessian_callback = HessianComputationCallback()
                    lr_monitor_callback = LearningRateMonitor(logging_interval="step")
                    plt_trainer_args["callbacks"] = [
                        checkpoint_callback,
                        #hessian_callback,
                        lr_monitor_callback,
                    ]
                    plt_trainer_args["logger"] = visualizer

                plugins = None
                plt_trainer_args["plugins"] = plugins

                wrapper_cls = get_model_wrapper(model_info, task)

                load_name = "../mase_output/vgg_cifar10_prune/software/transforms/prune/state_dict.pt"
                load_type = "pt"
                
                if load_name:
                    mask_collect = weight_masks
                    model = load_model(model_short_name, mask_collect, is_quantize, load_name, load_type=load_type, model=model)
                    logger.info(f"'{load_type}' checkpoint loaded before training")

                plt_trainer_args['accelerator'] = config['passes']['retrain']['trainer']['accelerator']
                plt_trainer_args['devices'] = config['passes']['retrain']['trainer']['devices']

                pl_model = wrapper_cls(
                    model,
                    dataset_info=dataset_info,
                    learning_rate = config['passes']['retrain']['training']['learning_rate'],
                    epochs = config['passes']['retrain']['training']['max_epochs'],
                    weight_decay = config['passes']['retrain']['training']['weight_decay'],
                    optimizer = config['passes']['retrain']['training']['optimizer'],
                    batch_size = config['passes']['retrain']['training']['batch_size'],
                )

                trainer = pl.Trainer(
                    **plt_trainer_args, 
                    max_epochs=config['passes']['retrain']['training']['max_epochs'], 
                )

                trainer.fit(
                    pl_model,
                    datamodule=data_module,
                )

                save_dir = retrain_save_path
                torch.save(pl_model.state_dict(), f"{save_dir}/model.ckpt")
                logger.info(f"model is successfully fine-tuned and saved to {save_dir}/model.ckpt!")

            case "huffman":
                is_huffman = config['passes']['huffman']['is_huffman']
                if is_huffman:
                    layer_huffman_info = PASSES["huffman"](pl_model, cf_args, model_info, data_module, task, accelerator, huffman_pass_config)
                    decoded_weights = PASSES["huffman_decode"](layer_huffman_info) 

            case "remove_prune_wrappers":
                # Removes the pruning-related hooks and makes pruning permanent
                graph, _ = PASSES[pass_name](graph, pass_args=None)
            case "conv_bn_fusion":
                graph, _ = PASSES[pass_name](graph, pass_args=None)
            case "logicnets_fusion":
                graph, _ = PASSES[pass_name](graph, pass_args=pass_config)
            case "onnx_annotate":
                onnx_dir = save_dir / "onnx"
                onnx_dir.mkdir(parents=True, exist_ok=True)
                kwargs = {
                    "save_path": onnx_dir,
                    "data_path": pass_config["data_path"],
                }
                graph, _ = PASSES[pass_name](graph, **kwargs)
            case _:
                my_pass = PASSES[pass_name]
                graph, _ = my_pass(graph, pass_args=pass_config)

        assert isinstance(
            graph, MaseGraph
        ), f"Return type of {pass_name} must be MaseGraph, got {type(graph)}"

    '''
    if save_dir is not None:
        transformed_ckpt = save_dir / "transformed_ckpt"
        transformed_ckpt.mkdir(parents=True, exist_ok=True)
        graph, _ = metadata_value_type_cast_transform_pass(
            graph, pass_args={"fn": to_numpy_if_tensor}
        )
        graph, _ = save_mase_graph_interface_pass(graph, pass_args=transformed_ckpt)
    return graph
    '''