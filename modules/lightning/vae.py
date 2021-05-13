import pyro
import torch
import pyro.distributions as dist
import pyro.contrib.examples.util
import pytorch_lightning as pl

from criterion.get_losses import get_losses
from modules.classifier import Classifier
from modules.decoder import Decoder
from modules.encoder import Encoder
from modules.vae import VAE
from utils.torch_utils import to_gpu


class LightningVAE(pl.LightningModule):
    def __init__(self, a, vae_config, optimizer_config):
        super(LightningVAE, self).__init__()
        # create the encoder and decoder networks
        self.vae = VAE(**vae_config)
        self.a = a
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.optimizer_config = optimizer_config

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.optimizer_config)


    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch["labelled"]
        x_unsupervised, _ = batch["missing"]
        loss_unsup, loss_sup, loss_class, y_logits = get_losses(self.vae, y, x, x_unsupervised)
        loss = loss_unsup + loss_sup + self.a * loss_class
        probs = torch.nn.functional.softmax(y_logits, dim=1)

        self.train_acc(probs, y)
        # TODO: ??
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_sup", loss_sup, on_step=False, on_epoch=True)
        self.log("train_loss_unsup", loss_unsup, on_step=False, on_epoch=True)
        self.log("train_loss_class", loss_class, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch

        loss_unsup, loss_sup, loss_class, y_logits = get_losses(self.vae, y, x, x)
        loss = loss_unsup + loss_sup + self.a * loss_class
        probs = torch.nn.functional.softmax(y_logits, dim=1)

        self.val_acc(probs, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_loss_sup", loss_sup, on_step=False, on_epoch=True)
        self.log("val_loss_unsup", loss_unsup, on_step=False, on_epoch=True)
        self.log("val_loss_class", loss_class, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)


