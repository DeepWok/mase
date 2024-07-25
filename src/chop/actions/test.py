import logging
import os
import pickle

import pytorch_lightning as pl
from chop.tools.plt_wrapper import get_model_wrapper
from chop.tools.checkpoint_load import load_model
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.skip(reason="This isn't a test")
def test(
    model: pl.LightningModule,
    model_info: dict,
    data_module: pl.LightningDataModule,
    dataset_info: dict,
    task: str,
    optimizer: str,
    learning_rate: float,
    weight_decay: float,
    plt_trainer_args: dict,
    auto_requeue: bool,
    save_path: str,
    visualizer: TensorBoardLogger,
    load_name: str,
    load_type: str,
):
    """
    Evaluate a trained model using PyTorch Lightning.

    Args:
        model (pl.LightningModule): Model to be trained.
        model_info (dict): Information about the model.
        data_module (pl.LightningDataModule): Data module for the model.
        dataset_info (dict): Information about the dataset.
        task (str): Task to be performed.
        optimizer (str): Optimizer to be used.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        plt_trainer_args (dict): Arguments for PyTorch Lightning Trainer.
        auto_requeue (bool): Requeue on SLURM.
        save_path (str): Path to save the model.
        visualizer (TensorBoardLogger): Tensorboard logger.
        load_name (str): Name of the checkpoint to load.
        load_type (str): Type of the checkpoint to load.

    """
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt_trainer_args["callbacks"] = []
        plt_trainer_args["logger"] = visualizer

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
        dataset_info=dataset_info,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer=optimizer,
    )

    trainer = pl.Trainer(**plt_trainer_args)
    if data_module.dataset_info.test_split_available:
        trainer.test(plt_model, datamodule=data_module)
    elif data_module.dataset_info.pred_split_available:
        predicted_results = trainer.predict(plt_model, datamodule=data_module)
        pred_save_name = os.path.join(save_path, "predicted_result.pkl")
        with open(pred_save_name, "wb") as f:
            pickle.dump(predicted_results, f)
        logger.info(f"Predicted results is saved to {pred_save_name}")
    else:
        raise ValueError(
            f"Test or pred split not available for dataset {data_module.info.name}"
        )
