import logging
import os
from pathlib import Path

import pytorch_lightning as pl
from chop.plt_wrapper import get_model_wrapper
from chop.tools.checkpoint_load import load_model
from chop.tools.get_input import get_dummy_input
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.passes.graph.transforms.interface import save_mase_graph_transform_pass
from chop.ir.graph import MaseGraph
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from torch.distributed.fsdp import FullyShardedDataParallel
from pytorch_lightning.strategies import DDPStrategy


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
        model = load_model(load_name, load_type=load_type, model=model)
        logger.info(f"'{load_type}' checkpoint loaded before training")

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
        graph = MaseGraph(model)
        dummy_input = get_dummy_input(model_info, data_module, task)
        graph = init_metadata_analysis_pass(graph, None)
        graph = add_common_metadata_analysis_pass(graph, dummy_input)
        graph = add_software_metadata_analysis_pass(graph, None)
        transformed_ckpt = Path(save_path) / "transformed_ckpt"
        transformed_ckpt.mkdir(parents=True, exist_ok=True)
        save_mase_graph_transform_pass(graph, pass_args=transformed_ckpt)
