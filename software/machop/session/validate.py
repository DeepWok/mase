import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from ..utils import load_pt_pl_or_pkl_checkpoint_into_pt_model
from .plt_wrapper import get_model_wrapper


def validate(
    model_name,
    info,
    model,
    task,
    data_module,
    plt_trainer_args,
    save_path,
    load_name,
    load_type,
):
    if save_path is not None:
        tb_logger = TensorBoardLogger(save_dir=save_path, name="logs_validate-sw")
        plt_trainer_args["logger"] = tb_logger

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
    plt_model = wrapper_cls(
        model,
        info=info,
    )

    trainer = pl.Trainer(**plt_trainer_args)
    trainer.validate(plt_model, datamodule=data_module)
