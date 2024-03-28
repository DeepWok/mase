#!/usr/bin/env python3
# NOTE: This is not really a test, but a script to just informally validate
# functionality via trial and error. Feel free to modify this file as needed.

import logging
import os
import sys
from pathlib import Path

import toml
import torch
import pytorch_lightning as pl
import pdb

# Housekeeping -------------------------------------------------------------------------
os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append(Path(__file__).resolve().parents[5].as_posix())

import chop.models as models
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    profile_statistics_analysis_pass,
    prune_transform_pass,
)
from chop.passes.graph.analysis.pruning.calculate_sparsity import (
    add_pruning_metadata_analysis_pass,
)

from chop.passes.graph import PASSES
from chop.ir.graph.mase_graph import MaseGraph
from chop.tools.get_input import InputGenerator, get_dummy_input
from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity
import pprint
from chop.passes.graph.utils import get_node_actual_target
import os
set_logging_verbosity("debug")

logger = logging.getLogger("chop.test")

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

from chop.passes.graph.transforms.pruning.pruning_methods import weight_criteria_map, activation_criteria_map

from chop.passes.graph.transforms import metadata_value_type_cast_transform_pass
from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target
from chop.plt_wrapper import get_model_wrapper
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import copy

pp = pprint.PrettyPrinter(indent=4)

if "test" in os.getcwd(): # if in "mase/machop/test/passes/graph/transforms/prune"
    config_file = "../../../../../configs/examples/vgg_cifar10_prune_retrain.toml"
elif "machop" in os.getcwd(): # if in "mase/machop"
    config_file = "configs/examples/vgg_cifar10_prune_retrain.toml"
elif "mase" in os.getcwd():
    config_file = "chop/configs/examples/vgg_cifar10_prune_retrain.toml"



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
    
def run(config_file):
    BATCH_SIZE = 512
    root = Path(__file__).resolve().parents[5]
    config = toml.load(open(config_file))
    with open(config_file) as f:
        config = toml.load(f)
        print("config: ");print(config)

    model_name = config['model']
    dataset_name = config['dataset']
    load_name = config['passes']['retrain']['load_name']
    load_type = config['passes']['retrain']['load_type']
    accelerator = config['passes']['retrain']['trainer']['accelerator']
    task = config['task']
    model_short_name = config['model'];print(model_short_name)
    dataset_short_name = config['dataset'];print(dataset_short_name)
    accelerator = parse_accelerator(accelerator)

    # if has pre-trained model, then load
    dataset_info = get_dataset_info(dataset_name)
    model = models.get_model(
    name=model_short_name,
    task=task,
    dataset_info=dataset_info,
    pretrained=False,
    checkpoint=None,
    quant_config=None,
    )
    weight_masks = None; is_quantize = False
    model = pre_transform_load(model_short_name, weight_masks, is_quantize, load_name=load_name, load_type=load_type, model=model)
    model.to(accelerator)

    save_dir = f"../mase_output/group3_test/"   
    prune_save_dir = os.path.join(save_dir, "prune")
    quantize_save_dir = os.path.join(save_dir, "quantize")
    retrain_save_dir = os.path.join(save_dir, "retrain")

    # concrete forward args for freezing dynamic control flow in forward pass
    model_info = models.get_model_info(model_short_name)
    if "cf_args" not in config:
        cf_args = get_cf_args(model_info=model_info, task=task, model=model)
    else:
        cf_args = config["cf_args"]


    # graph generation
    graph = MaseGraph(model=model, cf_args=cf_args)
    graph, _ = init_metadata_analysis_pass(graph, pass_args=None)

    # create or load metadata.parameters and mase_graph.model
    if load_name is not None and load_type == "mz":
        graph, _ = load_mase_graph_interface_pass(graph, pass_args=load_name)
    else:
        dataset_info = get_dataset_info(dataset_name)
        model_info = models.get_model_info(model_name)
        data_module = MaseDataModule(
            model_name=model_name,
            name=dataset_name,
            batch_size=BATCH_SIZE,
            num_workers=0,
            tokenizer=None,
            max_token_len=None,
        )
        data_module.prepare_data() ; data_module.setup()
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
                pass_config['default']['config']['name'] = None
                pass_config['by'] = 'type'
                graph, _ = PASSES["quantize"](graph, pass_args=pass_config)
                is_quantize = True

                for n in graph.nodes:
                    if isinstance(get_node_actual_target(n), torch.nn.modules.Conv2d): 
                        if 'mase' in n.meta:
                            quantized_weight = get_node_actual_target(n).w_quantizer(get_node_actual_target(n).weight)
                            graph.model.state_dict()['.'.join(n.name.rsplit('_', 1)) + ".weight"].copy_(quantized_weight)
                            print(f"There is quantization at {n.name}, mase_op: {get_mase_op(n)}")

                # save the quantized model
                save_dir = quantize_save_dir ; save_dir = Path(save_dir) ; save_dir.mkdir(parents=True, exist_ok=True)           
                graph, _ = metadata_value_type_cast_transform_pass(graph, pass_args={"fn": to_numpy_if_tensor})
                graph, _ = save_mase_graph_interface_pass(graph, pass_args=save_dir) 
                logger.info(f"model is successfully quantized and saved to {save_dir}!")
        
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

                # pruning process
                graph, _ = PASSES[pass_name](
                    graph,
                    batch_size,
                    pass_config,
                )  

                # calculate the pruning sparsity
                graph, sparsity_info, weight_masks, act_masks = PASSES["add_pruning_metadata"](
                    graph,
                    {"dummy_in": dummy_in, "add_value": False}
                )

                # weight pruning is of static process, where the weight mask for each layer remains during fine-tuning
                # activation pruning is of dynamic pruning, where the activation mask for each input batch )iteration) will be updated

                # if we want to force activation pruning to be static, we could run the following two lines to save activation masks
                #torch.save(act_masks, "act_masks.pth")
                #print("activation mask saved")
                
                pp.pprint(sparsity_info)
                del act_masks # reduce memory
                
                # calculate model size and FLOP
                

                # save the pruned model
                save_dir = prune_save_dir ; save_dir = Path(save_dir) ; save_dir.mkdir(parents=True, exist_ok=True)           
                graph, _ = metadata_value_type_cast_transform_pass(graph, pass_args={"fn": to_numpy_if_tensor})
                graph, _ = save_mase_graph_interface_pass(graph, pass_args=save_dir) 
                logger.info(f"model is successfully pruned and saved to {save_dir}!")
            
            case "retrain":
                from pytorch_lightning.callbacks import Callback
                class HessianComputationCallback(Callback):
                    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                        loss = outputs['loss']
                        named_parameters = list(pl_module.named_parameters())
                        name, param = named_parameters[1]
                        #pdb.set_trace()
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
                if retrain_save_dir is not None:
                    # if retrain_save_path is None, the model will not be saved
                    if not os.path.isdir(retrain_save_dir):
                        os.makedirs(retrain_save_dir)
                    checkpoint_callback = ModelCheckpoint(
                        save_top_k=1,
                        monitor="val_loss_epoch",
                        mode="min",
                        filename="best",
                        dirpath=retrain_save_dir,
                        save_last=True,
                    )
                    hessian_callback = HessianComputationCallback()
                    lr_monitor_callback = LearningRateMonitor(logging_interval="step")
                    plt_trainer_args["callbacks"] = [
                        checkpoint_callback,
                        #hessian_callback,
                        lr_monitor_callback,
                    ]

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

                save_dir = retrain_save_dir
                torch.save(pl_model.state_dict(), f"{save_dir}/model.ckpt")
                logger.info(f"model is successfully fine-tuned and saved to {save_dir}/model.ckpt!")

            case "huffman":
                is_huffman = config['passes']['huffman']['is_huffman']
                if is_huffman:
                    layer_huffman_info = PASSES["huffman"](pl_model, cf_args, model_info, data_module, task, accelerator, huffman_pass_config)
                    decoded_weights = PASSES["huffman_decode"](layer_huffman_info) 


run(config_file)