import os
import torch
import pytorch_lightning as pl

from .plt_wrapper import get_model_wrapper
from ..utils import load_model


def test(model_name, info, model, task, data_loader, plt_trainer_args, load_path):
    wrapper_cls = get_model_wrapper(model_name, task)
    plt_model = wrapper_cls(model, info=info)

    plt_model = load_model(plt_model=plt_model, load_path=load_path)
    trainer = pl.Trainer(**plt_trainer_args)
    trainer.test(plt_model, dataloaders=data_loader.test_dataloader)
