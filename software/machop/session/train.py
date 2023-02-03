from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from .wrapper import ModelWrapper


def train(
        model,
        data_loader,
        optimizer, 
        learning_rate, 
        plt_trainer_args, 
        save_path):
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="best",
        dirpath=save_path,
        save_last=True,
    )
    plt_trainer_args['callbacks'] = [checkpoint_callback]
    plt_model = ModelWrapper(
        model,
        learning_rate=learning_rate,
        epochs=plt_trainer_args['max_epochs'],
        optimizer=optimizer)
    trainer = pl.Trainer(**plt_trainer_args)
    trainer.fit(
        plt_model, 
        train_dataloaders=data_loader.train_dataloader, 
        val_dataloaders=data_loader.val_dataloader)