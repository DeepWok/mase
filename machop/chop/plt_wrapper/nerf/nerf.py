import torch
from collections import defaultdict
from torchmetrics import MeanMetric


from ..base import WrapperBase
from .losses import loss_dict
from .metrics import psnr
from .visualization import visualize_depth


class NeRFModelWrapper(WrapperBase):
    def __init__(
        self,
        model,
        dataset_info=None,
        learning_rate=5e-4,
        weight_decay=0,
        epochs=200,
        optimizer=None,
    ):
        super().__init__(
            model=model,
            dataset_info=dataset_info,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            optimizer=optimizer,
        )
        self.save_hyperparameters(dataset_info.nerf_config)

        self.loss = loss_dict["color"](coef=1)
        self.validation_step_outputs = []

        self.psnr_test = MeanMetric()

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        return self.model(rays)

    def training_step(self, batch, batch_idx):
        rays, rgbs = batch["rays"], batch["rgbs"]
        results = self(rays)
        loss = self.loss(results, rgbs)

        with torch.no_grad():
            typ = "fine" if "rgb_fine" in results else "coarse"
            psnr_ = psnr(results[f"rgb_{typ}"], rgbs)

        # self.log('lr', get_learning_rate(self.optimizer))
        self.log("train/loss", loss)
        self.log("train/psnr", psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        rays, rgbs = batch["rays"], batch["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        results = self(rays)
        log = {"val_loss": self.loss(results, rgbs)}
        typ = "fine" if "rgb_fine" in results else "coarse"

        if batch_idx == 0:
            W, H = self.hparams.img_wh
            img = (
                results[f"rgb_{typ}"].view(H, W, 3).permute(2, 0, 1).cpu()
            )  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            depth = visualize_depth(results[f"depth_{typ}"].view(H, W))  # (3, H, W)
            stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
            self.logger.experiment.add_images(
                "val/GT_pred_depth", stack, self.global_step
            )

        psnr_ = psnr(results[f"rgb_{typ}"], rgbs)
        log["val_psnr"] = psnr_

        self.loss_val.update(log["val_loss"])
        self.validation_step_outputs.append(log)

        return log

    def on_validation_epoch_end(self):
        mean_loss = torch.stack(
            [x["val_loss"] for x in self.validation_step_outputs]
        ).mean()
        mean_psnr = torch.stack(
            [x["val_psnr"] for x in self.validation_step_outputs]
        ).mean()
        self.validation_step_outputs.clear()  # free memory

        self.log("val/loss", mean_loss)
        self.log("val/psnr", mean_psnr, prog_bar=True)

        loss_epoch = self.loss_val.compute()
        self.log("val_loss_epoch", loss_epoch, prog_bar=True)

    def test_step(self, batch, batch_idx):
        rays, rgbs = batch["rays"], batch["rgbs"]
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        results = self(rays)
        loss = self.loss(results, rgbs)
        self.loss_test.update(loss)

        typ = "fine" if "rgb_fine" in results else "coarse"
        psnr_ = psnr(results[f"rgb_{typ}"], rgbs)
        self.psnr_test.update(psnr_)
        return loss

    def on_test_epoch_end(self):
        self.log("test_psnr_epoch", self.psnr_test, prog_bar=True)
        self.log("test_loss_epoch", self.loss_test, prog_bar=True)

    # def on_test_epoch_end(self):
    #     self.log("test_acc_epoch", self.acc_test, prog_bar=True)
    #     self.log("test_loss_epoch", self.loss_test, prog_bar=True)

    # def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
    #     input_ids = batch["input_ids"]
    #     attention_mask = batch["attention_mask"]
    #     outputs = self.forward(input_ids, attention_mask, labels=None)
    #     logits = outputs["logits"]
    #     pred_ids = torch.max(logits, dim=1)
    #     return {"batch_idx": batch_idx, "outputs": outputs, "pred_ids": pred_ids}
