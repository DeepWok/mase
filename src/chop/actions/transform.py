import os
from copy import deepcopy
from pathlib import Path
import logging

import torch
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
from chop.passes.module import PASSES as MODULE_PASSES

logger = logging.getLogger(__name__)


def pre_transform_load(load_name: str, load_type: str, model: torch.nn.Module):
    if load_name is not None and load_type in ["pt", "pl"]:
        model = load_model(load_name=load_name, load_type=load_type, model=model)
    return model


def transform(
    model: torch.nn.Module,
    model_info,
    model_name,
    data_module,
    task: str,
    config: str,
    save_dir: str = None,
    load_name: str = None,
    load_type: str = None,
    accelerator: str = "auto",
):
    accelerator = parse_accelerator(accelerator)
    model = pre_transform_load(load_name=load_name, load_type=load_type, model=model)
    model.to(accelerator)

    config = load_config(config)
    transform_config = config["transform"]
    style = transform_config.get("style", "graph")
    if style == "graph":
        transform_graph(
            model=model,
            model_info=model_info,
            model_name=model_name,
            data_module=data_module,
            task=task,
            config=config,
            save_dir=save_dir,
            load_name=load_name,
            load_type=load_type,
            accelerator=accelerator,
        )
    elif style == "module":
        transform_module(
            model=model,
            model_info=model_info,
            model_name=model_name,
            data_module=data_module,
            task=task,
            config=config,
            save_dir=save_dir,
            load_name=load_name,
            load_type=load_type,
            accelerator=accelerator,
        )
    else:
        raise ValueError(f"Style {style} is not supported!")


def transform_module(
    model: torch.nn.Module,
    model_info,
    model_name,
    data_module,
    task: str,
    config: str,
    save_dir: str = None,
    load_name: str = None,
    load_type: str = None,
    accelerator: str = "auto",
):
    accelerator = parse_accelerator(accelerator)
    model = pre_transform_load(load_name=load_name, load_type=load_type, model=model)
    model.to(accelerator)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if load_name is not None:
        model = load_model(load_name, load_type=load_type, model=model)
        logger.info(f"'{load_type}' checkpoint loaded before training")

    pass_config = config["passes"]

    for pass_name, pass_config in pass_config.items():
        pass_name: str
        pass_config: dict
        match pass_name:
            case _:
                my_pass = MODULE_PASSES[pass_name]
                model, _ = my_pass(model, pass_args=pass_config)

    if save_dir is not None:
        transformed_ckpt = save_dir / "transformed_ckpt"
        state_dict_ckpt = os.path.join(transformed_ckpt, "state_dict.pt")
        transformed_ckpt.mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        torch.save(state_dict, state_dict_ckpt)
        logger.info(f"model saved at {state_dict_ckpt}")
    return model


def transform_graph(
    model: torch.nn.Module,
    model_info,
    model_name,
    data_module,
    task: str,
    config: str,
    save_dir: str = None,
    load_name: str = None,
    load_type: str = None,
    accelerator: str = "auto",
):
    accelerator = parse_accelerator(accelerator)
    model = pre_transform_load(load_name=load_name, load_type=load_type, model=model)
    model.to(accelerator)
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

    # create or load metadata.parameters and mase_graph.model
    if load_name is not None and load_type == "mz":
        graph, _ = load_mase_graph_interface_pass(
            graph, pass_args={"load_dir": load_name}
        )
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

    passes_config = config["passes"]
    for pass_name, pass_config in passes_config.items():
        pass_name: str
        pass_config: dict
        match pass_name:
            case "tensorrt":
                graph, _ = metadata_value_type_cast_transform_pass(
                    graph, pass_args={"fn": to_numpy_if_tensor}
                )
                ori_graph = deepcopy_mase_graph(graph)
                pass_save_dir = save_dir / "tensorrt"

                pass_config["task"] = task
                pass_config["dataset"] = config["dataset"]
                pass_config["batch_size"] = config["batch_size"]
                pass_config["model"] = config["model"]
                pass_config["data_module"] = data_module
                pass_config["accelerator"] = accelerator.type
                if accelerator.type == "cuda":
                    # TODO this seems innefective - known issue - https://github.com/NVIDIA/TensorRT/issues/2468
                    os.environ["CUDA_MODULE_LOADING"] = "LAZY"

                # Firstly fake quantize the model for calibration (only if using int8 precision otherwise skipped)
                graph, _ = PASSES["tensorrt_fake_quantize"](
                    graph, pass_args=pass_config
                )

                # Summarize to show what has been quantized
                PASSES["summarize_quantization"](
                    graph, {"save_dir": pass_save_dir, "original_graph": ori_graph}
                )

                # Then calibrate the model using the fake quantization to set AMAXs
                graph, _ = PASSES["tensorrt_calibrate"](graph, pass_args=pass_config)

                # Apply post-quantization fine tuning (Quantization Aware Training)
                graph, _ = PASSES["tensorrt_fine_tune"](graph, pass_args=pass_config)

                # Apply fp16 or layer-wise mixed precision quantization if necessary and convert the model to TensorRT format
                graph, runtime_meta = PASSES["tensorrt"](graph, pass_args=pass_config)

                # Perform runtime analysis on original and new graph
                _, _ = PASSES["runtime_analysis"](ori_graph, pass_args=pass_config)

                _, _ = PASSES["runtime_analysis"](
                    runtime_meta["trt_engine_path"], pass_args=pass_config
                )

            case "onnxruntime":
                pass_save_dir = save_dir / "onnxruntime"
                graph, _ = metadata_value_type_cast_transform_pass(
                    graph, pass_args={"fn": to_numpy_if_tensor}
                )
                ori_graph = deepcopy_mase_graph(graph)
                pass_config["data_module"] = data_module

                # crop the train dataloader to behave as the calibrated dataloader
                pass_config["data_module"].train_dataloader

                pass_config["task"] = task
                pass_config["accelerator"] = accelerator.type
                pass_config["batch_size"] = config["batch_size"]
                pass_config["model"] = config["model"]
                pass_config["dataset"] = config["dataset"]

                if accelerator.type == "cuda":
                    # TODO this seems innefective - known issue - https://github.com/NVIDIA/TensorRT/issues/2468
                    os.environ["CUDA_MODULE_LOADING"] = "LAZY"

                graph, runtime_meta = PASSES["onnxruntime"](
                    graph, pass_args=pass_config
                )

                # if user has set runtime_anaylsis, run the runtime analysis pass
                if "runtime_analysis" not in pass_config:
                    break

                # Extract the 'runtime_analysis' dictionary by stripping the config
                runtime_analysis = pass_config.pop("runtime_analysis", {})
                pass_config.update(runtime_analysis)

                original_graph_analysis = pass_config.get(
                    "original_graph_analysis", True
                )
                if original_graph_analysis:
                    logger.info("Performing runtime analysis on original graph...")
                    _, _ = PASSES["runtime_analysis"](ori_graph, pass_args=pass_config)
                optimized_graph_analysis = pass_config.get(
                    "optimized_graph_analysis", True
                )
                if optimized_graph_analysis:
                    logger.info(
                        "Performing runtime analysis on onnx-optimized graph..."
                    )
                    _, _ = PASSES["runtime_analysis"](
                        runtime_meta["onnx_path"], pass_args=pass_config
                    )

                # Peform runtime analysis on quantized forms if appropriate
                quantized_graph_analysis = pass_config.get(
                    "quantized_graph_analysis", True
                )
                if quantized_graph_analysis:
                    try:
                        quant_types = pass_config["default"]["config"]["quantize_types"]
                    except KeyError:
                        quant_types = []
                    for quant_type in quant_types:
                        match quant_type:
                            case "static":
                                logger.info(
                                    "Performing runtime analysis on static quantized graph..."
                                )
                                _, _ = PASSES["runtime_analysis"](
                                    runtime_meta["onnx_static_quantized_path"],
                                    pass_args=pass_config,
                                )
                            case "dynamic":
                                logger.info(
                                    "Performing runtime analysis on dynamic quantized graph..."
                                )
                                _, _ = PASSES["runtime_analysis"](
                                    runtime_meta["onnx_dynamic_quantized_path"],
                                    pass_args=pass_config,
                                )
                            case "auto":
                                logger.info(
                                    "Performing runtime analysis on auto mixed precision quantized graph..."
                                )
                                _, _ = PASSES["runtime_analysis"](
                                    runtime_meta["onnx_auto_mixed_precision_path"],
                                    pass_args=pass_config,
                                )

            case "quantize":
                pass_save_dir = save_dir / "quantize"
                graph, _ = metadata_value_type_cast_transform_pass(
                    graph, pass_args={"fn": to_numpy_if_tensor}
                )
                ori_graph = deepcopy_mase_graph(graph)
                graph, _ = PASSES["quantize"](graph, pass_args=pass_config)
                PASSES["summarize_quantization"](
                    graph, {"save_dir": pass_save_dir, "original_graph": ori_graph}
                )
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
                    "save_path",
                    save_dir / "save_node_meta_param" / "node_meta_param.toml",
                )
                # TODO: fix me
                # to save the meta parameters of the nodes,
                # we have to run this cast,
                # because current meta parameters contains tensors
                # but this cast is not inveritble
                # if there are other passes after "save_node_meta_param"
                # relying on the tensor/numpy values in the meta parameters
                # the transform/analysis will fail
                graph, _ = metadata_value_type_cast_transform_pass(
                    graph, pass_args={"fn": to_numpy_if_tensor}
                )
                graph, _ = PASSES[pass_name](graph, pass_args=pass_save_path)
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
                pass_config["model_name"] = model_name
                pass_config["input_generator"] = input_generator
                prune_save_dir = save_dir / "prune"
                prune_save_dir.mkdir(parents=True, exist_ok=True)
                graph, _ = PASSES[pass_name](
                    graph,
                    save_dir=prune_save_dir,
                    config=pass_config,
                )
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

    if save_dir is not None:
        transformed_ckpt = save_dir / "transformed_ckpt"
        transformed_ckpt.mkdir(parents=True, exist_ok=True)
        graph, _ = metadata_value_type_cast_transform_pass(
            graph, pass_args={"fn": to_numpy_if_tensor}
        )
        save_mase_graph_interface_pass(graph, pass_args=transformed_ckpt)
    return graph
