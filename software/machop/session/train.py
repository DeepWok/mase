import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

from ..utils import load_pt_pl_or_pkl_checkpoint_into_pt_model
from .plt_wrapper import get_model_wrapper

logger = logging.getLogger(__name__)


def train(
    model_name,
    info,
    model,
    task,
    data_module,
    optimizer,
    learning_rate,
    plt_trainer_args,
    auto_requeue,
    save_path,
    load_name,
    load_type,
):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    # if save_path is None, the model will not be saved
    if save_path is not None:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            filename="best",
            dirpath=save_path,
            save_last=True,
        )
        tb_logger = TensorBoardLogger(save_dir=save_path, name="logs")
        gpu_usage_callback = DeviceStatsMonitor()
        lr_monitor_callback = LearningRateMonitor(logging_interval="step")
        plt_trainer_args["callbacks"] = [
            checkpoint_callback,
            gpu_usage_callback,
            lr_monitor_callback,
        ]
        plt_trainer_args["logger"] = tb_logger

    # plugin
    if auto_requeue:
        plugins = [SLURMEnvironment(auto_requeue=auto_requeue)]
    else:
        plugins = None
    plt_trainer_args["plugins"] = plugins

    # Check optimizer
    if plt_trainer_args["strategy"] in ["deepspeed_stage_3"]:
        assert optimizer in [
            "FusedAdam",
            "fused_adam",
        ], "optimizer should be 'fused_adam' given --strategy={}".format(
            plt_trainer_args["strategy"]
        )

    wrapper_cls = get_model_wrapper(model_name, task)

    if load_name is not None:
        if isinstance(model, dict):
            model["model"] = load_pt_pl_or_pkl_checkpoint_into_pt_model(
                load_name=load_name, load_type=load_type, model=model["model"]
            )

        else:
            model = load_pt_pl_or_pkl_checkpoint_into_pt_model(
                load_name=load_name, load_type=load_type, model=model
            )
        logger.info(f"'{load_type}' checkpoint loaded before training")

    pl_model = wrapper_cls(
        model,
        info=info,
        learning_rate=learning_rate,
        epochs=plt_trainer_args["max_epochs"],
        optimizer=optimizer,
    )

    trainer = pl.Trainer(**plt_trainer_args)
    trainer.fit(
        pl_model,
        datamodule=data_module,
    )
