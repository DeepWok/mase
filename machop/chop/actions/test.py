import logging
import os
import pickle

import pytorch_lightning as pl
from chop.plt_wrapper import get_model_wrapper
from chop.tools.checkpoint_load import load_model
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.skip(reason="This isn't a test")
def test(
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
