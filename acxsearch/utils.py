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
import pickle
import json
import os
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

def init_dataset(dataset_name, batch_size, model_name):
    dataset_info = get_dataset_info(dataset_name)
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
    return data_module

def initialize_graph(model_name, dataset_name, batch_size, load_name, load_type):
    task = "classification"
    model_info = get_model_info(model_name)
    dataset_info = get_dataset_info(dataset_name)
    model = get_model(
        name=model_name,
        task=task,
        dataset_info=dataset_info,
        checkpoint=model_name,
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


def save_accuracy_list(acc_list_all, directory='saved_results', base_filename='acc_list_all'):
    """
    Save the accuracy list to both pickle and JSON formats.
    
    Args:
        acc_list_all: List of accuracy results to save
        directory: Directory to save the files in (default: 'saved_results')
        base_filename: Base name for the saved files (default: 'acc_list_all')
    
    Returns:
        tuple: Paths to the saved pickle and JSON files
    """
    # Create a directory for saved results if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Define file paths
    pickle_path = os.path.join(directory, f"{base_filename}.pkl")
    json_path = os.path.join(directory, f"{base_filename}.json")
    
    # Save using pickle (binary format)
    with open(pickle_path, 'wb') as f:
        pickle.dump(acc_list_all, f)
    
    # Save using JSON (human-readable format)
    # Convert the data to a format that can be serialized to JSON
    json_compatible_data = []
    for acc_list in acc_list_all:
        json_compatible_data.append([(int(x), float(y)) for x, y in acc_list])
    
    with open(json_path, 'w') as f:
        json.dump(json_compatible_data, f, indent=4)
    
    print(f"Saved accuracy lists to {pickle_path} and {json_path}")
    return pickle_path, json_path


def plot_accuracy_vs_bitwidth(acc_list, 
                            break_points=None,  # [(y_min1, y_max1, x_min1, x_max1), (y_min2, y_max2, x_min2, x_max2)]
                            highlight_region=None,  # (y_lower, y_upper)
                            title="Accuracy vs Bit Width",
                            marker_style='bo-',
                            figsize=(10, 6),
                            labels=None,
                            x_range=None):  # Added x_range parameter
    """
    Plot accuracy vs bit width with two charts: one showing the full range and 
    another highlighting a specific region.
    
    Args:
        acc_list: List of tuples [(bit_width, accuracy), ...] or list of such lists
        break_points: List of tuples [(y_min1, y_max1, x_min1, x_max1), (y_min2, y_max2, x_min2, x_max2)]
        highlight_region: Tuple of (y_lower, y_upper) to mark with dashed lines
        title: Plot title
        marker_style: Style of plot markers
        figsize: Figure size
        labels: List of labels for each accuracy list
        x_range: Tuple of (x_min, x_max) to set the x-axis range
    """
    import matplotlib.pyplot as plt
    
    # Check if acc_list is a list of lists
    if acc_list and isinstance(acc_list[0], list):
        # Multiple accuracy lists
        all_data = []
        for sublist in acc_list:
            if sublist:  # Check if sublist is not empty
                bit_widths, accuracies = zip(*sublist)
                all_data.append((bit_widths, accuracies))
    else:
        # Single accuracy list
        bit_widths, accuracies = zip(*acc_list)
        all_data = [(bit_widths, accuracies)]
    
    # Create figure with white background
    plt.style.use('default')
    f = plt.figure(figsize=figsize)
    
    # Create two subplots with height ratio 2:1
    gs = f.add_gridspec(3, 1)
    ax1 = f.add_subplot(gs[0:2, 0])  # First 2/3 of the height
    ax2 = f.add_subplot(gs[2, 0])    # Last 1/3 of the height
    
    # Plot data on both axes
    for i, (bit_widths, accuracies) in enumerate(all_data):
        # Use different colors for multiple lists
        if len(all_data) > 1:
            current_style = marker_style.replace('b', f'C{i}')
        else:
            current_style = marker_style
            
        # Use label if provided
        label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
        
        # Plot on the first axis (full view)
        ax1.plot(bit_widths, accuracies, current_style, linewidth=1, markersize=4, label=label)
        
        # Plot on the second axis (zoomed/highlighted view)
        ax2.plot(bit_widths, accuracies, current_style, linewidth=1, markersize=4, label=label)
    
    # Customize legend appearance
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.edgecolor'] = 'gray'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.shadow'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    
    # Set x-axis range if specified
    if x_range:
        ax1.set_xlim(x_range)
        ax2.set_xlim(x_range)
    
    # Set axis limits for both charts
    if break_points and len(break_points) > 0:
        y_min1, y_max1, x_min1, x_max1 = break_points[0]
        ax1.set_ylim(y_min1, y_max1)
        ax1.set_xlim(x_min1, x_max1)
    
    if break_points and len(break_points) > 1:
        y_min2, y_max2, x_min2, x_max2 = break_points[1]
        ax2.set_ylim(y_min2, y_max2)
        ax2.set_xlim(x_min2, x_max2)
    
    # Add dashed lines for highlight region if specified
    if highlight_region:
        y_lower, y_upper = highlight_region
        ax1.axhline(y=y_lower, color='r', linestyle='--', alpha=0.5)
        ax1.axhline(y=y_upper, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=y_lower, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=y_upper, color='r', linestyle='--', alpha=0.5)
    
    # Customize grid and labels
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle=':', alpha=0.4)
    
    # Set x-ticks
    if all_data:
        all_bit_widths = sorted(set([bw for bws, _ in all_data for bw in bws]))
        ax1.set_xticks(all_bit_widths)
        ax2.set_xticks(all_bit_widths)
    
    # Set labels
    ax1.set_title("Full Range View", fontsize=8)
    ax2.set_title("Highlighted Region", fontsize=8)
    ax2.set_xlabel('Bit Width', fontsize=10)
    f.text(0.00, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize=10)
    plt.suptitle(title, fontsize=14, y=0.98)
    
    # Add legend if we have multiple data series
    if len(all_data) > 1:
        ax1.legend(loc='upper right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return f
