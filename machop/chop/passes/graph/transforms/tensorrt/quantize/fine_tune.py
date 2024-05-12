import logging
import torch
import logging
import os
from .utils import prepare_save_path, check_for_value_in_dict


def tensorrt_fine_tune_transform_pass(graph, pass_args=None):
    """
    Fine-tunes a quantized model using Quantization Aware Training (QAT) to improve its accuracy post-quantization.

    This pass employs a fine-tuning process that adjusts the quantized model's weights in a way that acknowledges the quantization effects, thereby aiming to recover or even surpass the original model's accuracy. The training process uses a reduced number of epochs and a significantly lower learning rate compared to the initial training phase, following a cosine annealing learning rate schedule.

    :param graph: The model graph to be fine-tuned. This graph should already be quantized.
    :type graph: MaseGraph
    :param pass_args: A dictionary containing arguments for fine-tuning, such as the number of epochs (`epochs`), the initial learning rate (`initial_learning_rate`), and the final learning rate (`final_learning_rate`). These parameters allow customization of the training regime based on the specific needs of the model and dataset.
    :type pass_args: dict, optional
    :return: A tuple containing the fine-tuned graph and an empty dictionary. The empty dictionary is a placeholder for potential extensions.
    :rtype: tuple(MaseGraph, dict)

    The default training regime involves:
    - Using 10% of the original training epochs.
    - Starting with 1% of the original training learning rate.
    - Employing a cosine annealing schedule to reduce the learning rate to 0.01% of the initial training learning rate by the end of fine-tuning.

    The resulting fine-tuned model checkpoints are saved in the following directory structure, facilitating easy access and version control:

    - mase_output
        - tensorrt
            - quantization
                - model_task_dataset_date
                    - cache
                    - ckpts
                        - fine_tuning
                    - json
                    - onnx
                    - trt

    Example of usage:

        graph = MaseGraph(...)
        fine_tuned_graph, _ = tensorrt_fine_tune_transform_pass(graph, {'epochs': 5, 'initial_learning_rate': 0.001, 'final_learning_rate': 0.00001})

    This example demonstrates initiating the fine-tuning process with custom epochs, and initial and final learning rates, adapting the training regime to the specific requirements of the quantized model.
    """
    trainer = FineTuning(graph, pass_args)
    ckpt = trainer.train()

    # Link the model with the graph for further operations or evaluations
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)

    return graph, {"ckpt_save_path": ckpt}


class FineTuning:
    def __init__(self, graph, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.graph = graph
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.graph.model.to(self.device)

    def train(self):
        """
        For QAT it is typical to employ 10% of the original training epochs,
        starting at 1% of the initial training learning rate, and a cosine
        annealing learning rate schedule that follows the decreasing half of
        a cosine period, down to 1% of the initial fine tuning learning rate
        (0.01% of the initial training learning rate). However this default
        can be overidden by setting the `epochs`, `initial_learning_rate` and
        `final_learning_rate` in `passes.tensorrt.fine_tune`.
        """
        if not self.config.get("fine_tune", {}).get("fine_tune", True):
            self.logger.warning(
                "Fine tuning is disabled in the config. Skipping QAT fine tuning."
            )
            return None

        if not check_for_value_in_dict(self.config, "int8"):
            self.logger.warning(
                "int8 precision not found in config. Skipping QAT fine tuning."
            )
            return None

        from chop.actions import train
        from chop.models import get_model_info

        # load the settings and default to chop default parameters
        model_info = get_model_info(self.config["data_module"].model_name)
        weight_decay = (
            self.config["weight_decay"] if "weight_decay" in self.config else 0
        )
        optimizer = self.config["optimizer"] if "optimizer" in self.config else "adam"

        # Check if user would like to override the initial learning rate otherwise default to 1% of original LR
        try:
            initial_fine_tune_lr = (self.config["initial_learning_rate"]) * 0.01
        except KeyError:
            initial_fine_tune_lr = (self.config.get("learning_rate", 1e-5)) * 0.01

        # Check if user would like to override the final learning rate otherwise default to
        # 1% of initial learning rate or 0.01% of original learning rate
        try:
            eta_min = self.config["final_learning_rate"]
        except KeyError:
            eta_min = initial_fine_tune_lr * 0.01  # Decreases to

        # Check if user would like to override the number of epochs otherwise default to 10% of original epochs
        try:
            epochs = self.config["fine_tune"]["epochs"]
        except KeyError:
            epochs = int(self.config.get("max_epochs", 20) * 0.1)

        t_max = int(len(self.config["data_module"].train_dataloader()) * epochs)

        ckpt_save_path = prepare_save_path(
            self.config, method="ckpts/fine_tuning", suffix="ckpt"
        )

        scheduler_args = {"t_max": t_max, "eta_min": eta_min}

        plt_trainer_args = {
            "max_epochs": epochs,
            "accelerator": self.config["accelerator"],
        }
        self.logger.info(f"Starting Fine Tuning for {epochs} epochs...")
        train(
            self.graph.model,
            model_info,
            self.config["data_module"],
            self.config["data_module"].dataset_info,
            "Quantization Fine Tuning",
            optimizer,
            initial_fine_tune_lr,
            weight_decay,
            scheduler_args,
            plt_trainer_args,
            False,
            ckpt_save_path,
            None,
            None,
            "",
        )

        self.logger.info("Fine Tuning Complete")
        return ckpt_save_path / "best.ckpt"
