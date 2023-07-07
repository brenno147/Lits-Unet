from torch import optim
from util import IoULoss
import segmentation_models_pytorch as smp
import torch
import lightning.pytorch as pl
import torchmetrics

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.lossFunc = smp.losses.JaccardLoss(mode='multiclass', classes=[1,2])
        self.trainIou = torchmetrics.JaccardIndex(task='multiclass', num_classes=3, average=None)
        self.testIou = torchmetrics.JaccardIndex(task='multiclass', num_classes=3, average=None)
        self.validIou = torchmetrics.JaccardIndex(task='multiclass', num_classes=3, average=None)

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        pred = self.model(x)

        loss = self.lossFunc(pred, y.long())
        # self.log("train_loss", loss, prog_bar=True)

        listIou = self.trainIou(pred, y.long())

        values = {"loss": loss, "train_bg_iou": listIou[0], "train_liver_iou": listIou[1], "train_tumor_iou": listIou[2]}
        self.log_dict(values, prog_bar=True)

        return values

    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        pred = self.model(x)
        y = y.type(torch.uint8)

        loss = self.lossFunc(pred, y.long())
        # self.log("test_loss", loss, prog_bar=True)

        listIou = self.validIou(pred, y.long())

        values = {"test_loss": loss, "test_bg_iou": listIou[0], "test_liver_iou": listIou[1], "test_tumor_iou": listIou[2]}
        self.log_dict(values)

    def validation_step(self, batch, batch_idx):
    # this is the validation loop
        x, y = batch
        pred = self.model(x)
        y = y.type(torch.uint8)

        loss = self.lossFunc(pred, y.long())
        # self.log("test_loss", loss, prog_bar=True)

        listIou = self.testIou(pred, y.long())

        values = {"valid_loss": loss, "valid_bg_iou": listIou[0], "valid_liver_iou": listIou[1], "valid_tumor_iou": listIou[2]}
        self.log_dict(values)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer