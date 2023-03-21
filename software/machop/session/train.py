import logging

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from ..utils import load_pt_pl_or_pkl_checkpoint_into_pt_model
from .plt_wrapper import get_model_wrapper

logger = logging.getLogger(__name__)


def train(
    model_name,
    info,
    model,
    task,
    data_loader,
    optimizer,
    learning_rate,
    plt_trainer_args,
    save_path,
    load_name,
    load_type,
):
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
        plt_trainer_args["callbacks"] = [checkpoint_callback]
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
        logger.info(f"'{load_type}' checkpoint loaded before training")
    # breakpoint()
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
        train_dataloaders=data_loader.train_dataloader,
        val_dataloaders=data_loader.val_dataloader,
    )
