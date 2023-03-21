import pytorch_lightning as pl

from ..utils import load_pt_pl_or_pkl_checkpoint_into_pt_model
from .plt_wrapper import get_model_wrapper


def test(
    model_name, info, model, task, data_loader, plt_trainer_args, load_name, load_type
):
    wrapper_cls = get_model_wrapper(model_name, task)

    if load_name is not None:
        model = load_pt_pl_or_pkl_checkpoint_into_pt_model(
            load_name=load_name, load_type=load_type, model=model
        )
    plt_model = wrapper_cls(
        model,
        info=info,
    )

    trainer = pl.Trainer(**plt_trainer_args)
    trainer.test(plt_model, dataloaders=data_loader.test_dataloader)
