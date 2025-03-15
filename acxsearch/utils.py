import pytorch_lightning as pl
from chop.tools.plt_wrapper import get_model_wrapper
from pathlib import Path
import torch
from chop.passes.graph.analysis import (
    add_common_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
    init_metadata_analysis_pass,
)
from chop.passes.graph.transforms import (
    quantize_transform_pass,
)
import time
from chop.tools.checkpoint_load import load_model
from chop.dataset import get_dataset_info, MaseDataModule
from chop.models import get_model, get_model_info
from chop.tools.get_input import get_dummy_input
from chop.ir.graph.mase_graph import MaseGraph
from tqdm import tqdm
import sys
import toml
import copy

sys.path.append(Path(__file__).resolve().parents[2].as_posix())
sys.path.append(Path(__file__).resolve().parents[3].as_posix())

def fine_tune(mg, ft_args):
    plt_trainer_args = {
        "devices": 1,
        "num_nodes": 1,
        "accelerator": "auto",
        "strategy": "ddp",
        "precision": "32",
        "callbacks": [],
        "plugins": None,
        "max_epochs": 1,
    }

    wrapper_cls = get_model_wrapper(ft_args["model_info"], ft_args["task"])
    # initialize mode
    pl_model = wrapper_cls(
        mg.model,
        dataset_info=ft_args["dataset_info"],
        learning_rate=5e-6,
        weight_decay=1e-5,
        epochs=plt_trainer_args["max_epochs"],
        optimizer="adamw",
    )
    trainer = pl.Trainer(**plt_trainer_args)
    trainer.fit(
        pl_model,
        datamodule=ft_args["data_module"],
    )
    current_loss = loss_cal(mg, ft_args["data_module"], ft_args["num_batchs"])

    return current_loss


def initialize_graph(model_name, dataset_name, batch_size, load_name, load_type):
    task = "classification"
    model_info = get_model_info(model_name)
    dataset_info = get_dataset_info(dataset_name)
    model = get_model(
        name=model_name,
        task=task,
        dataset_info=dataset_info,
        checkpoint=None,
        pretrained=True,
    )

    if load_name is not None:
        model = load_model(load_name, load_type=load_type, model=model)
    data_module = MaseDataModule(
        name=dataset_name,
        batch_size=batch_size,
        num_workers=56,
        tokenizer=None,
        max_token_len=512,
        load_from_cache_file=True,
        model_name=model_name,
    )
    data_module.prepare_data()
    data_module.setup()

    dummy_in = get_dummy_input(
        model_info=model_info,
        data_module=data_module,
        task=task,
        device=next(model.parameters()).device,
    )

    mg = MaseGraph(model)
    mg, _ = init_metadata_analysis_pass(mg, None)
    mg, _ = add_common_metadata_analysis_pass(
        mg, {"dummy_in": dummy_in, "add_value": False}
    )

    return_meta = {
        "data_module": data_module,
        "dummy_in": dummy_in,
        "model_info": model_info,
        "dataset_info": dataset_info,
    }
    return mg, return_meta


def loss_cal(
    model, test_loader, max_iteration=None, description=None, accelerator="cuda"
):
    i = 0
    losses = []
    model = model.to(accelerator)
    max_iteration = len(test_loader) if max_iteration is None else max_iteration
    with torch.no_grad():
        q = tqdm(test_loader, desc=description)
        for inputs in q:
            xs, ys = inputs
            preds = model(xs.to(accelerator))
            loss = torch.nn.functional.cross_entropy(preds, ys.to(accelerator))
            losses.append(loss)
            i += 1
            if i >= max_iteration:
                break
    loss_avg = sum(losses) / len(losses)
    loss = float(loss_avg)
    print(loss)
    return loss


def acc_cal(
    model, test_loader, max_iteration=None, description=None, accelerator="cuda"
):
    pos = 0
    tot = 0
    i = 0
    model = model.to(accelerator)
    max_iteration = len(test_loader) if max_iteration is None else max_iteration
    with torch.no_grad():
        q = tqdm(test_loader, desc=description)
        for inp, target in q:
            i += 1
            inp = inp.to(accelerator)
            target = target.to(accelerator)
            out = model(inp)
            pos_num = torch.sum(out.argmax(1) == target).item()
            pos += pos_num
            tot += inp.size(0)
            q.set_postfix({"acc": pos / tot})
            if i >= max_iteration:
                break
    print(pos / tot)
    return pos / tot


def load_config(config_path):
    """Load from a toml config file and convert "NA" to None."""
    with open(config_path, "r") as f:
        config = toml.load(f)
    # config = convert_str_na_to_none(config)
    return config


def parse_config_choice(config_choice: dict):
    def dynamic_loops(elements, depth, new_list=[], current=[]):
        if depth == 0:
            new_list.append(current)
            return new_list
        for element in elements[len(elements)-depth]:
            dynamic_loops(elements, depth - 1, new_list, current + [element])
        return new_list
    for key, value in config_choice.items():
        depth = len(value)
        new_list = dynamic_loops(value, depth)
        config_choice[key] = new_list
