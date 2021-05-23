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
from utils.model_utils import build_dense_nn
from utils.torch_utils import to_gpu


class LightningClassifier(pl.LightningModule):
    def __init__(self, classifier_config, optimizer_config):
        super(LightningClassifier, self).__init__()
        # create the encoder and decoder networks
        self.classifier = Classifier(**classifier_config)
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.optimizer_config = optimizer_config
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.optimizer_config)


    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch["labelled"]
        y_logits = self.classifier(x)
        loss = self.cross_entropy(y_logits, y)
        probs = torch.nn.functional.softmax(y_logits, dim=1)

        self.train_acc(probs, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch

        y_logits = self.classifier(x)
        loss = self.cross_entropy(y_logits, y)
        probs = torch.nn.functional.softmax(y_logits, dim=1)

        self.val_acc(probs, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)


