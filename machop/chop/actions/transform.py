import os
from copy import deepcopy
from pathlib import Path
import logging
import pprint
import torch
from chop.passes.graph import PASSES
from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    init_metadata_analysis_pass,
    profile_statistics_analysis_pass,
    add_pruning_metadata_analysis_pass,
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
from chop.tools.utils import device

from chop.actions import train, test
import torchvision
import torchvision.transforms as transforms
import heapq
import collections

from chop.passes.graph.transforms import metadata_value_type_cast_transform_pass
from chop.passes.module import PASSES as MODULE_PASSES


logger = logging.getLogger(__name__)
pp = pprint.PrettyPrinter(indent=4)


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
                model = my_pass(model, pass_args=pass_config)

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
    dataset_info,
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

    graph = MaseGraph(model=model, cf_args=cf_args)
    graph, _ = init_metadata_analysis_pass(graph, pass_args=None)

    if load_name is not None and load_type == "mz":
        graph = load_mase_graph_interface_pass(graph, pass_args=load_name)
    else:
        dummy_in = get_dummy_input(
            model_info=model_info,
            data_module=data_module,
            task=task,
            device=device,
        )
        if len(graph.model.additional_inputs) > 0:
            dummy_in = dummy_in | graph.model.additional_inputs
        graph, _ = add_common_metadata_analysis_pass(
            graph, pass_args={"dummy_in": dummy_in, "force_device_meta": False}
        )
        graph, _ = add_software_metadata_analysis_pass(graph, pass_args=None)

    pass_config = config["passes"]

    for pass_name, pass_config in pass_config.items():
        pass_name: str
        pass_config: dict
        match pass_name:
            case "profile_statistics":
                input_generator = InputGenerator(
                    model_info=model_info,
                    data_module=data_module,
                    task=task,
                    which_dataloader="train",
                )
                pass_config["input_generator"] = input_generator
                graph = PASSES[pass_name](graph, pass_args=pass_config)
            case "report_graph":
                pass_file_name = pass_config.get(
                    "file_name", save_dir / "report_graph.txt"
                )
                graph = PASSES[pass_name](graph, file_name=pass_file_name)
            case "report_node_type":
                graph = PASSES[pass_name](graph, pass_args=None)
            case "report_node_meta_param":
                pass_save_path = pass_config.get("save_path", save_dir / "report")
                pass_config["save_path"] = pass_save_path
                graph = PASSES[pass_name](graph, pass_args=pass_config)
            case "report_node_shape":
                graph = PASSES[pass_name](graph, pass_args=None)
            case "report_node_type":
                graph = PASSES[pass_name](graph, pass_args=None)
            case "report_node_hardware_type":
                graph = PASSES[pass_name](graph, pass_args=None)
            case "report_node_shape":
                graph = PASSES[pass_name](graph, pass_args=None)
            case "report_node_type":
                graph = PASSES[pass_name](graph, pass_args=None)
            case "load_mase_graph":
                pass_load_dir = pass_config["load_dir"]
                graph = PASSES[pass_name](graph, pass_args=pass_load_dir)
            case "load_node_meta_param":
                pass_load_path = pass_config["load_path"]
                graph = PASSES[pass_name](graph, pass_args=pass_load_path)
            case "save_mase_graph":
                pass_save_dir = pass_config.get(
                    "save_dir", save_dir / "saved_mase_graph"
                )
                graph = PASSES[pass_name](graph, pass_args=pass_save_dir)
            case "save_node_meta_param":
                pass_save_path = pass_config.get(
                    "save_path", save_dir / "saved_node_meta_param"
                )
                graph = PASSES[pass_name](graph, pass_args=pass_save_path)
            case "prune":
                input_generator = InputGenerator(
                    model_info=model_info,
                    data_module=data_module,
                    task=task,
                    which_dataloader="val",
                )
                dummy_input = get_dummy_input(
                    model_info, data_module, task=task, device="cpu"
                )
                pass_config["model_name"] = model_name
                pass_config["input_generator"] = input_generator
                pass_config["dummy_in"] = dummy_input
                graph = PASSES[pass_name](
                    graph,
                    pass_args=pass_config,
                )
                graph.model.to("cuda")
                graph, sparsity_info = add_pruning_metadata_analysis_pass(
                    graph, {"dummy_in": dummy_input, "add_value": False}
                )
                pp.pprint(sparsity_info)
                """
                print('Start Encoding')
                # Apply Huffman Encoding to the pruned weights
                class HuffmanEncoder:
                    def __init__(self, weights):
                        self.weights = weights
                        self.encoded_values = {}
                        self.build_huffman_tree()

                    def build_huffman_tree(self):
                        freq = collections.Counter(self.weights)
                        heap = [[weight, [value, ""]] for value, weight in freq.items()]
                        heapq.heapify(heap)
                        while len(heap) > 1:
                            lo = heapq.heappop(heap)
                            hi = heapq.heappop(heap)
                            for pair in lo[1:]:
                                pair[1] = '0' + pair[1]
                            for pair in hi[1:]:
                                pair[1] = '1' + pair[1]
                            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
                        encoding = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
                        self.encoded_values = {value: code for value, code in encoding}

                    def encode(self, value):
                        return self.encoded_values[value]

                    def decode(self, encoded_value):
                        reverse_encoded_values = {v: k for k, v in self.encoded_values.items()}
                        decoded_value = ""
                        current_code = ""
                        for bit in encoded_value:
                            current_code += bit
                            if current_code in reverse_encoded_values:
                                decoded_value += reverse_encoded_values[current_code]
                                current_code = ""
                        return decoded_value
                
                encoded_list = []
                for name, param in model.named_parameters():
                    if 'original' in name:
                        pruned_weights = param.detach().cpu().numpy().flatten()
                        huffman_encoder = HuffmanEncoder(pruned_weights)
                        encoded_weights = [huffman_encoder.encode(w) for w in pruned_weights]
                        encoded_list.append(encoded_weights)

                if save_dir is not None:
                    transformed_ckpt = save_dir / f"{pass_name}_ckpt"
                    transformed_ckpt.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'state_dict': model.state_dict(),
                        'encoded_weights': encoded_weights
                    }, os.path.join(transformed_ckpt, "state_dict_with_huffman.pt"))
                print('End Encoding')
                """

                if save_dir is not None:
                    transformed_ckpt = save_dir / f"{pass_name}_ckpt"
                    transformed_ckpt.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        model.state_dict(),
                        os.path.join(transformed_ckpt, "state_dict.pt"),
                    )
                return graph

            case "retrain":
                input_generator = InputGenerator(
                    model_info=model_info,
                    data_module=data_module,
                    task=task,
                    which_dataloader="val",
                )
                dummy_input = get_dummy_input(
                    model_info, data_module, task=task, device="cpu"
                )

                pass_config["model_name"] = model_name
                pass_config["input_generator"] = input_generator
                pass_config["dummy_in"] = dummy_input

                max_epoch = pass_config["config"]["epoch"]
                batch_size = pass_config["config"]["batch_size"]
                lr = pass_config["config"]["learning_rate"]

                module = torch.load(load_name)

                def get_parametrized_layers(module):
                    parametrized_layers = []
                    for name, weights in module.items():
                        if "original" in name:
                            parametrized_layers.append(weights)
                    return parametrized_layers

                def load_pruned_weights(model, layers, index):
                    for _, layer in model.named_children():
                        if isinstance(
                            layer,
                            (
                                torch.nn.modules.conv.Conv2d,
                                torch.nn.modules.linear.Linear,
                            ),
                        ):
                            layer.weight.data.copy_(layers[index[0]])
                            index[0] += 1
                        elif isinstance(layer, torch.nn.Module):
                            load_pruned_weights(layer, layers, index)
                    return model

                if dataset_info.name == "cifar10":
                    transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]
                    )
                    trainset = torchvision.datasets.CIFAR10(
                        root="./data", train=True, download=True, transform=transform
                    )
                    trainloader = torch.utils.data.DataLoader(
                        trainset, batch_size=batch_size, shuffle=True, num_workers=2
                    )

                elif dataset_info.name == "imagenet":
                    transform = transforms.Compose(
                        [
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    )
                    trainset = torchvision.datasets.ImageNet(
                        root="./data", train=True, download=True, transform=transform
                    )
                    trainloader = torch.utils.data.DataLoader(
                        trainset, batch_size=batch_size, shuffle=True, num_workers=2
                    )

                else:
                    raise ValueError(
                        "Dataset {} is not supported for pruning yet".format(
                            dataset_info.name
                        )
                    )

                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                parametrized_layers = get_parametrized_layers(module)
                index = [0]
                pruned_model = load_pruned_weights(model, parametrized_layers, index)

                print("Start retraining the model")
                num_epochs = max_epoch
                for epoch in range(num_epochs):
                    running_loss = 0.0
                    for i, data in enumerate(trainloader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()

                        outputs = pruned_model(inputs.to("cuda"))
                        loss = criterion(outputs, labels.to("cuda"))
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                        if i % 100 == 99:  # Print every 100 mini-batches
                            print(
                                "[Num of epoch:%d, %5dth batch] loss: %.3f"
                                % (epoch + 1, i + 1, running_loss / 100)
                            )
                            running_loss = 0.0
                print("Retrain complete")
                if save_dir is not None:
                    transformed_ckpt = save_dir / f"{pass_name}_ckpt"
                    transformed_ckpt.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        pruned_model,
                        os.path.join(transformed_ckpt, "retrain_model.ckpt"),
                    )

                return graph
            case "remove_prune_wrappers":
                # Removes the pruning-related hooks and makes pruning permanent
                graph = PASSES[pass_name](graph, pass_args=None)
            case "conv_bn_fusion":
                graph = PASSES[pass_name](graph, pass_args=None)
            case "logicnets_fusion":
                graph = PASSES[pass_name](graph, pass_args=pass_config)
            case "onnx_annotate":
                onnx_dir = save_dir / "onnx"
                onnx_dir.mkdir(parents=True, exist_ok=True)
                kwargs = {
                    "save_path": onnx_dir,
                    "data_path": pass_config["data_path"],
                }
                graph = PASSES[pass_name](graph, **kwargs)
            case _:
                my_pass = PASSES[pass_name]
                graph = my_pass(graph, pass_args=pass_config)
        ## graph, pass_info = graph
        assert isinstance(
            graph, MaseGraph
        ), f"Return type of {pass_name} must be MaseGraph, got {type(graph)}"

    if save_dir is not None:
        transformed_ckpt = save_dir / f"{pass_name}_ckpt"
        transformed_ckpt.mkdir(parents=True, exist_ok=True)
        save_mase_graph_interface_pass(graph, pass_args=transformed_ckpt)
    return graph
