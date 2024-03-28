import os
from copy import deepcopy
from pathlib import Path
import logging
import copy

import torch
import pickle
from chop.passes.graph import PASSES

import pytorch_lightning as pl
from chop.plt_wrapper import get_model_wrapper
from chop.tools.checkpoint_load import load_model
from chop.tools.get_input import get_dummy_input
from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target

from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    init_metadata_analysis_pass,
)


from chop.passes.graph.interface import save_mase_graph_interface_pass
from chop.ir.graph import MaseGraph
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from torch.distributed.fsdp import FullyShardedDataParallel
from pytorch_lightning.strategies import DDPStrategy
from chop.tools.config_load import load_config

from chop.tools.get_input import InputGenerator, get_cf_args, get_dummy_input


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
import pprint
import pdb

import gc
gc.collect()

import torch.nn.utils.prune as prune
from chop.passes.graph.utils import get_node_actual_target


global act_masks
act_masks = None

logger = logging.getLogger(__name__)

pp = pprint.PrettyPrinter(indent=4)

# test

def pre_transform_load(mask, is_quantize, load_name: str, load_type: str, model: torch.nn.Module):
    if load_name is not None and load_type in ["pt", "pl"]:
        model = load_model(mask, is_quantize, load_name=load_name, load_type=load_type, model=model)
    return model


def prune_and_retrain(
    model: torch.nn.Module,
    model_info,
    model_name,
    data_module,
    dataset_info,
    task,
    config,
    visualizer,
    prune_save_dir: str = None,
    retrain_save_path: str=None,
    load_name: str = None,
    load_type: str = None,
    accelerator: str = "auto",
):
    is_quantize = False
    accelerator = parse_accelerator(accelerator)
    config = load_config(config)
    load_name = config['retrain']['load_name']
    load_type = config['retrain']['load_type']
    
    is_huffman = config['passes']['huffman']['is_huffman']
    print("Huffman?: ", is_huffman)

    mask=None
    model = pre_transform_load(mask, is_quantize, load_name=load_name, load_type=load_type, model=model)
    model.to(accelerator)
    
    save_dir = prune_save_dir
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # concrete forward args for freezing dynamic control flow in forward pass
    if "cf_args" not in config:
        cf_args = get_cf_args(model_info=model_info, task=task, model=model)
    else:
        cf_args = config["cf_args"]

    # graph generation
    graph = MaseGraph(model=model, cf_args=cf_args)
    # graph_metadata = Mase
    graph, _ = init_metadata_analysis_pass(graph, pass_args=None)

    # logger.info(f"graph: {graph.fx_graph}")

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
            case "prune":
                # NOTE: The input generator is only used for when the user wants to
                # enforce or observe activation sparsity. Otherwise, it's ignored.
                # We use the validation dataloader as that doesn't shuffle the input
                # data. This determinism helps establish a fair ground in draw
                # layer-wise comparisons between activation pruning strategies.
                input_generator = InputGenerator(
                    model_info=model_info,
                    data_module=data_module,
                    task=task,
                    which_dataloader="val",
                )
                print("pass_config: ") ; print(pass_config)
                pass_config["model_name"] = model_name
                pass_config["input_generator"] = input_generator
                #prune_save_dir = save_dir / "prune"
                #prune_save_dir.mkdir(parents=True, exist_ok=True)
                batch_size = config['retrain']['training']['batch_size']
                graph, _ = PASSES[pass_name](
                    graph,
                    batch_size, # self_added
                    pass_config,
                )
                #graph, sparsity_info, mask_collect, act_masks = PASSES["add_pruning_metadata"](
                graph, sparsity_info, mask_collect = PASSES["add_pruning_metadata"](
                    graph,
                    {"dummy_in": dummy_in, "add_value": False}
                )
                #torch.save(act_masks, "/mnt/d/imperial/second_term/adls/projects/mase/machop/act_masks.pth")
                #print("activation mask saved")
                #pp.pprint(sparsity_info)

                #del act_masks # to save memory

            case "quantize":
                gc.collect()
                graph, _ = metadata_value_type_cast_transform_pass(graph, pass_args={"fn": to_numpy_if_tensor})
                #ori_mg = deepcopy_mase_graph(graph)
                graph, _ = PASSES["quantize"](graph, pass_args=pass_config)
                #PASSES["summarize_quantization"](ori_mg, graph, save_dir="quantize_summary")
                # nn = [n for n in graph.fx_graph.nodes]
                # from get_node_actual_target(n).weight to get_node_actual_target(n).w_quantizer(get_node_actual_target(n).weight)
                is_quantize = True

                for n in graph.nodes:
                    if isinstance(get_node_actual_target(n), torch.nn.modules.Conv2d): 
                        if 'mase' in n.meta:
                            quantized_weight = get_node_actual_target(n).w_quantizer(get_node_actual_target(n).weight)
                            graph.model.state_dict()['.'.join(n.name.rsplit('_', 1)) + ".weight"].copy_(quantized_weight)
                            print(f"There is quantization at {n.name}, mase_op: {get_mase_op(n)}")


        assert isinstance(
            graph, MaseGraph
        ), f"Return type of {pass_name} must be MaseGraph, got {type(graph)}"

    if save_dir is not None:
        # pdb.set_trace()
        transformed_ckpt = save_dir / "transformed_ckpt"
        transformed_ckpt.mkdir(parents=True, exist_ok=True)
        graph, _ = metadata_value_type_cast_transform_pass(
            graph, pass_args={"fn": to_numpy_if_tensor}
        )
        graph, _ = save_mase_graph_interface_pass(graph, pass_args=transformed_ckpt) 
        # save the pruned model

    ###############################
    #re-train
    ###############################
        
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

    #load_name = "/mnt/d/imperial/second_term/adls/projects/mase/mase_output/vgg_cifar10_prune/software/prune/transformed_ckpt/state_dict.pt"
    load_name = "/content/prune_mase/mase_output/vgg_cifar10_prune/software/prune/transformed_ckpt/state_dict.pt"
    load_type = "pt"
    #pdb.set_trace()
    
    if load_name is not None:
        model = load_model(mask_collect, is_quantize, load_name, load_type=load_type, model=model)
        #model = load_model(load_name, load_type=load_type, model=model)
        logger.info(f"'{load_type}' checkpoint loaded before training")


    plt_trainer_args['accelerator'] = config['retrain']['trainer']['accelerator']
    plt_trainer_args['devices'] = config['retrain']['trainer']['devices']

    pl_model = wrapper_cls(
        model,
        dataset_info=dataset_info,
        learning_rate = config['retrain']['training']['learning_rate'],
        epochs = config['retrain']['training']['max_epochs'],
        weight_decay = config['retrain']['training']['weight_decay'],
        optimizer = config['retrain']['training']['optimizer'],
        batch_size = config['retrain']['training']['batch_size'],
    )

    trainer = pl.Trainer(
        **plt_trainer_args, 
        max_epochs=config['retrain']['training']['max_epochs'], 
        # limit_train_batches=3  only 3 batches per epoch
    )

    trainer.fit(
        pl_model,
        datamodule=data_module,
    )

    torch.save(pl_model.state_dict(), "chop/model.ckpt")


    if is_huffman:
        layer_huffman_info = PASSES["huffman"](pl_model, cf_args, model_info, data_module, task, accelerator, huffman_pass_config)
        decoded_weights = PASSES["huffman_decode"](layer_huffman_info)
        #print(decoded_weights)
    

    