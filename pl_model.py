import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset import ListDataset
from model import YOLOv4

from lars import LARS
from ranger import Ranger
from radam import RAdam
from sched_del import DelayedCosineAnnealingLR
import config.params as hparams
torch.backends.cudnn.benchmark = True


class YOLOv4PL(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.train_ds = ListDataset(hparams.train_ds, train=True, img_extensions=hparams.img_extensions)
        self.valid_ds = ListDataset(hparams.valid_ds, train=False, img_extensions=hparams.img_extensions)

        self.model = YOLOv4(n_classes=hparams.n_classes,
                            pretrained=hparams.pretrained,
                            dropblock=hparams.Dropblock,
                            sam=hparams.SAM,
                            eca=hparams.ECA,
                            ws=hparams.WS,
                            iou_aware=hparams.iou_aware,
                            coord=hparams.coord,
                            hard_mish=hparams.hard_mish,
                            asff=hparams.asff,
                            repulsion_loss=hparams.repulsion_loss,
                            acff=hparams.acff,
                            bcn=hparams.bcn,
                            mbn=hparams.mbn).cuda()

    def train_dataloader(self):
        train_dl = DataLoader(self.train_ds, batch_size=hparams.bs, collate_fn=self.train_ds.collate_fn,
                              pin_memory=True, num_workers=hparams.n_cpu)
        return train_dl

    def val_dataloader(self):
        valid_dl = DataLoader(self.valid_ds, batch_size=hparams.bs, collate_fn=self.valid_ds.collate_fn,
                              pin_memory=True, num_workers=hparams.n_cpu)
        return valid_dl

    def forward(self, x, y=None):
        return self.model(x, y)

    def basic_training_step(self, batch):
        filenames, images, labels = batch
        y_hat, loss = self(images, labels)
        logger_logs = {"training_loss": loss}

        return {"training_loss": loss, "loss": loss, "log": logger_logs}

    def sat_fgsm_training_step(self, batch, epsilon=0.01):
        filenames, images, labels = batch

        images.requires_grad_(True)
        y_hat, loss = self(images, labels)
        loss.backward()
        data_grad = images.grad.data
        images.requires_grad_(False)
        images = torch.clamp(images + data_grad.sign() * epsilon, 0, 1)
        return self.basic_training_step((filenames, images, labels))

    def sat_vanila_training_step(self, batch, epsilon=1):
        filenames, images, labels = batch

        images.requires_grad_(True)
        y_hat, loss = self(images, labels)
        loss.backward()
        data_grad = images.grad.data
        images.requires_grad_(False)
        images = torch.clamp(images + data_grad, 0, 1)
        return self.basic_training_step((filenames, images, labels))

    def training_step(self, batch, batch_idx):
        if hparams.SAT == "vanila":
            return self.sat_vanila_training_step(batch, hparams.epsilon)
        elif hparams.SAT == "fgsm":
            return self.sat_fgsm_training_step(batch, hparams.epsilon)
        else:
            return self.basic_training_step(batch)

    def training_epoch_end(self, outputs):
        training_loss_mean = torch.stack([x['training_loss'] for x in outputs]).mean()
        return {"loss": training_loss_mean, "log": {"training_loss_epoch": training_loss_mean}}

    def validation_step(self, batch, batch_idx):
        filenames, images, labels = batch
        y_hat, loss = self(images, labels)
        logger_logs = {"training_loss": loss}
        return {"val_loss": loss, "log": logger_logs}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        logger_logs = {"validation_loss": val_loss_mean}

        return {"val_loss": val_loss_mean, "log": logger_logs}

    def configure_optimizers(self):
        # With this thing we get only params, which requires grad (weights needed to train)
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if hparams.optimizer == "Ranger":
            self.optimizer = Ranger(params, hparams.lr, weight_decay=hparams.wd)
        elif hparams.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(params, hparams.lr, momentum=hparams.momentum,
                                             weight_decay=hparams.wd)
        elif hparams.optimizer == "LARS":
            self.optimizer = LARS(params, lr=hparams.lr, momentum=hparams.momentum,
                                  weight_decay=hparams.wd, max_epoch=hparams.epochs)
        elif hparams.optimizer == "RAdam":
            self.optimizer = RAdam(params, lr=hparams.lr, weight_decay=hparams.wd)

        if hparams.scheduler == "Cosine Warm-up":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, hparams.lr,
                                                                 epochs=hparams.epochs, steps_per_epoch=1,
                                                                 pct_start=hparams.pct_start)
        if hparams.scheduler == "Cosine Delayed":
            self.scheduler = DelayedCosineAnnealingLR(self.optimizer, hparams.flat_epochs,
                                                      hparams.cosine_epochs)

        sched_dict = {'scheduler': self.scheduler}

        return [self.optimizer], [sched_dict]