import logging
import os
from pathlib import Path

import pytorch_lightning as pl
from chop.plt_wrapper import get_model_wrapper
from chop.tools.checkpoint_load import (
    load_model,
    reapply_parametrizations_from_state_dict,
    load_state_dict,
    reappply_activations,
    reapply_parametrizations_mg_module,
)
from chop.models import get_model_info, get_model
from chop.tools.get_input import get_dummy_input, InputGenerator
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.passes.graph.interface import (
    save_mase_graph_interface_pass,
    save_pruned_train_model,
)
from chop.passes.graph.transforms import metadata_value_type_cast_transform_pass
from chop.ir.graph import MaseGraph
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from torch.distributed.fsdp import FullyShardedDataParallel
from pytorch_lightning.strategies import DDPStrategy
from chop.tools.utils import parse_accelerator, to_numpy_if_tensor


logger = logging.getLogger(__name__)


# class CustomFSDPStrategy(DDPStrategy):
#     def configure_ddp(self):
#         # model = DistributedDataParallel(model)
#         fsdp_model = FullyShardedDataParallel(
#             self.model,
#             # fsdp_auto_wrap_policy=default_auto_wrap_policy,
#             # cpu_offload=CPUOffload(offload_params=True),
#         )
#         self.model = fsdp_model


def train(
    model,
    model_info,
    data_module,
    dataset_info,
    task,
    optimizer,
    learning_rate,
    weight_decay,
    plt_trainer_args,
    auto_requeue,
    save_path,
    visualizer,
    load_name,
    load_type,
):
    if save_path is not None:
        # if save_path is None, the model will not be saved
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss_epoch",
            mode="min",
            filename="best",
            dirpath=save_path,
            save_last=True,
        )
        # tb_logger = TensorBoardLogger(save_dir=save_path, name="logs")
        lr_monitor_callback = LearningRateMonitor(logging_interval="step")
        plt_trainer_args["callbacks"] = [
            checkpoint_callback,
            lr_monitor_callback,
        ]
        plt_trainer_args["logger"] = visualizer

    # plugin
    if auto_requeue:
        plugins = [SLURMEnvironment(auto_requeue=auto_requeue)]
    else:
        plugins = None
    plt_trainer_args["plugins"] = plugins

    # Check optimizer
    # if plt_trainer_args["strategy"] in ["deepspeed_stage_3"]:
    #     assert optimizer in [
    #         "FusedAdam",
    #         "fused_adam",
    #     ], "optimizer should be 'fused_adam' given --strategy={}".format(
    #         plt_trainer_args["strategy"]
    #     )
    # elif plt_trainer_args["strategy"] in ["fsdp_custom"]:
    #     plt_trainer_args["strategy"] = CustomFSDPStrategy()

    wrapper_cls = get_model_wrapper(model_info, task)

    if load_name is not None:
        if load_type != "mz":

            model = load_model(load_name, load_type=load_type, model=model)
            logger.info(f"'{load_type}' checkpoint loaded before training")
            state_dict = load_state_dict(load_name, load_type)
            activation_config = None
            print("model_info", type(model_info))
            print("model_info", model_info.name)
            if "activations" in state_dict.keys():
                data_module.prepare_data()
                data_module.setup()

                input_generator = InputGenerator(
                    data_module=data_module,
                    model_info=model_info,
                    task=task,
                    which_dataloader="train",
                    max_batches=1,
                )

                dummy_in = next(iter(input_generator))
                _ = model(**dummy_in)
                graph = MaseGraph(model)
                graph, _ = init_metadata_analysis_pass(graph, None)
                graph, _ = add_common_metadata_analysis_pass(
                    graph, {"dummy_in": dummy_in}
                )
                graph, _ = add_software_metadata_analysis_pass(graph, None)

                activation_config = state_dict["activations"]
                state_dict = state_dict["state_dict"]
                # reapply activation parametrisations
                print("activations_Applied")
                reappply_activations(graph, activation_config)
                reapply_parametrizations_mg_module(graph, state_dict)
                model = graph.model
            else:
                reapply_parametrizations_from_state_dict(model, state_dict)
    pl_model = wrapper_cls(
        model,
        dataset_info=dataset_info,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=plt_trainer_args["max_epochs"],
        optimizer=optimizer,
    )

    trainer = pl.Trainer(**plt_trainer_args)

    trainer.fit(
        pl_model,
        datamodule=data_module,
    )

    # Save the trained model along with relevant metadata in the training_ckpts folder.
    # NOTE: This is important if the model was previously transformed with architectural
    # changes. The state dictionary that's saved by PyTorch Lightning wouldn't work.
    if save_path is not None and load_name is not None and load_type == "mz":
        accelerator = plt_trainer_args["accelerator"]
        accelerator = parse_accelerator(accelerator)
        graph = MaseGraph(model)
        dummy_input = get_dummy_input(model_info, data_module, task, device=accelerator)
        graph, _ = init_metadata_analysis_pass(graph, None)
        graph, _ = add_common_metadata_analysis_pass(graph, {"dummy_in": dummy_input})
        graph, _ = add_software_metadata_analysis_pass(graph, None)
        transformed_ckpt = Path(save_path) / "transformed_ckpt"
        transformed_ckpt.mkdir(parents=True, exist_ok=True)
        graph, _ = metadata_value_type_cast_transform_pass(
            graph, pass_args={"fn": to_numpy_if_tensor}
        )
        save_mase_graph_interface_pass(graph, pass_args=transformed_ckpt)
        print("T")

    if save_path is not None and load_name is not None and load_type == "pt":
        transformed_ckpt = Path(save_path) / "transformed_ckpt"
        transformed_ckpt.mkdir(parents=True, exist_ok=True)
        save_pruned_train_model(model, transformed_ckpt, activation_config)
