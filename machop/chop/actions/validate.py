import os

import pytorch_lightning as pl
from chop.plt_wrapper import get_model_wrapper
from chop.tools.checkpoint_load import load_model
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment


def validate(
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
    if save_path is not None:
        tb_logger = TensorBoardLogger(save_dir=save_path, name="logs_validate-sw")
        plt_trainer_args["callbacks"] = []
        plt_trainer_args["logger"] = tb_logger

    # plugin
    if auto_requeue:
        plugins = [SLURMEnvironment(auto_requeue=auto_requeue)]
    else:
        plugins = None
    plt_trainer_args["plugins"] = plugins

    wrapper_cls = get_model_wrapper(model_name, task)

    if load_name is not None and load_type in ["pt", "pl", "mz"]:
        if isinstance(model, dict):
            model["model"] = load_model(
                load_name, load_type=load_type, model=model["model"]
            )
        else:
            model = load_model(load_name, load_type=load_type, model=model)
    plt_model = wrapper_cls(
        model, info=info, learning_rate=learning_rate, optimizer=optimizer
    )

    trainer = pl.Trainer(**plt_trainer_args)
    trainer.validate(plt_model, datamodule=data_module)
