import torch
import torch.nn as nn

from .base import VisionModelWrapper
from .losses import v8SegmentationLoss


class UltralyticsSegmentationWrapper(VisionModelWrapper):
    def __init__(
        self,
        model,
        dataset_info,
        learning_rate=1e-4,
        weight_decay=0,
        scheduler_args=None,
        epochs=3,
        criterion=v8SegmentationLoss,
        optimizer=None,
    ):
        super().__init__(
            model=model,
            dataset_info=dataset_info,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler_args=scheduler_args,
            epochs=epochs,
            optimizer=optimizer,
        )
        self.model = model

        # Need to pass model.args to the model
        self.criterion = criterion(
            model=self.model,
            box=7.5,
            cls=0.5,
            dfl=1.5,
        )  # Overrides the WrapperBase loss_fn (BCE)
        # Uses base hyperparameters from YOLO Info

    def training_step(self, batch, batch_idx):
        inputs = batch["img"]
        outputs = self.forward(inputs)

        loss, _ = self.criterion(outputs, batch)
        self.log("loss", loss.detach().item())
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["img"]
        outputs = self.forward(inputs)
        loss, _ = self.criterion(outputs, batch)
        self.log("val_loss", loss.detach().item())
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch["img"]
        outputs = self.forward(inputs)

        loss, _ = self.criterion(outputs, batch)
        self.log("test_loss", loss.detach().item())
        return loss

    def on_test_epoch_end(self):
        print("Test: ", self.loss_test)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        prediction = self.model.predict(batch["img"])
        return prediction
