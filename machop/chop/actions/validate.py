import os
import pickle
import logging

import pytorch_lightning as pl
from chop.plt_wrapper import get_model_wrapper
from chop.tools.checkpoint_load import load_model
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

logger = logging.getLogger(__name__)


def validate(
    model,
    tokenizer,
    model_info,
    data_module,
    dataset_info,
    task,
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
    if save_path is not None:
        tb_logger = TensorBoardLogger(save_dir=save_path, name="logs_test-sw")
        plt_trainer_args["callbacks"] = []
        plt_trainer_args["logger"] = tb_logger

    # plugin
    if auto_requeue:
        plugins = [SLURMEnvironment(auto_requeue=auto_requeue)]
    else:
        plugins = None
    plt_trainer_args["plugins"] = plugins

    wrapper_cls = get_model_wrapper(model_info, task)

    if load_name is not None:
        model = load_model(load_name, load_type=load_type, model=model)
    plt_model = wrapper_cls(
        model,
        tokenizer=tokenizer,
        info=dataset_info,
        learning_rate=learning_rate,
        optimizer=optimizer,
    )

    trainer = pl.Trainer(**plt_trainer_args)

    if data_module.dataset_info.validation_split_available:
        trainer.validate(plt_model, datamodule=data_module)
    else:
        logger.warning(
            f"Validation split not available for dataset {data_module.info.name}"
        )
