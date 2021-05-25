import torch

import pytorch_lightning as pl
from torch.nn.functional import binary_cross_entropy, cross_entropy, kl_div
from torch.distributions import normal, kl


from modules.classifier import Classifier
from modules.decoder import Decoder
from modules.encoder import Encoder
from utils.torch_utils import to_gpu


class LightningVAE(pl.LightningModule):
    def __init__(self, a, vae_config, optimizer_config, use_cuda=False):
        super(LightningVAE, self).__init__()
        # create the encoder and decoder networks
        # self.vae = VAE(**vae_config)
        self.use_cuda = use_cuda
        self.z_dim = vae_config['z_dim']

        # create the encoder and decoder networks
        self.encoder = Encoder(self.z_dim,  **vae_config['encoder_config'])
        self.classifier = Classifier(**vae_config['classifier_config'])
        self.decoder = Decoder(self.z_dim, **vae_config['decoder_config'])


        if self.use_cuda:
            self.cuda()

        self.kl_weight = 1
        self.a = a
        self.use_a = False
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.optimizer_config = optimizer_config


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.optimizer_config)


    def training_step(self, batch, batch_idx): # pylint: disable=unused-argument
        x, y = batch['labelled']
        x_unsupervised, _ = batch["missing"]
        loss_class, loss_s_recon, loss_s_KL, y_logits = self.supervised_step(x, y)
        loss_u_recon, loss_u_KL, y_entropy = self.unsupervised_step(x_unsupervised)
        loss = loss_s_recon + loss_u_recon + self.kl_weight * (loss_s_KL + loss_u_KL - y_entropy) + self.a * loss_class
        
        # probs = torch.nn.functional.softmax(y_logits, dim=1)

        self.train_acc(torch.argmax(y_logits,dim=1), y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_sup_recon", loss_s_recon, on_step=False, on_epoch=True)
        self.log("train_sup_KL", loss_s_KL, on_step=False, on_epoch=True)
        self.log("train_unsup_recon", loss_u_recon, on_step=False, on_epoch=True)
        self.log("train_unsup_KL", loss_u_KL, on_step=False, on_epoch=True)
        self.log("train_loss_class", loss_class, on_step=False, on_epoch=True)
        self.log("train_y_entropy", y_entropy, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        loss_class, loss_s_recon, loss_s_KL, y_logits = self.supervised_step(x, y)
        loss_u_recon, loss_u_KL, y_entropy = self.unsupervised_step(x)
        loss = loss_s_recon 
        loss += loss_u_recon 
        loss += self.kl_weight * (loss_s_KL + loss_u_KL - y_entropy)
        loss += self.a * loss_class

        probs = torch.nn.functional.softmax(y_logits, dim=1)

        self.val_acc(probs, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_sup_recon", loss_s_recon, on_step=False, on_epoch=True)
        self.log("val_sup_KL", loss_s_KL, on_step=False, on_epoch=True)
        self.log("val_unsup_recon", loss_u_recon, on_step=False, on_epoch=True)
        self.log("val_unsup_KL", loss_u_KL, on_step=False, on_epoch=True)
        self.log("val_loss_class", loss_class, on_step=False, on_epoch=True)
        self.log("val_y_entropy", y_entropy, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)


    def supervised_step(self, x, y):
        if self.encoder.use_y:
            mu, var = self.encoder(x, y)
        else:
            mu, var = self.encoder(x)
        y_dist = self.classifier(x) # <- only for class loss)
        loss_class = cross_entropy(y_dist, y, reduction='mean')
        z, kl_div = self.sample_z(mu, var)
        loss_recon = self.recon_loss(y, x, z)
        return loss_class, loss_recon.mean(), kl_div.mean(), y_dist


    def _unsupervised_step_y(self, x):

        # unsupervised
        y_logits = self.classifier(x)
        y_dist = torch.nn.functional.softmax(y_logits, dim=-1)
        y_log_dist = torch.nn.functional.log_softmax(y_logits, dim=-1)
        y_entropy = -(y_dist * y_log_dist).sum(dim=-1).mean()

        total_loss_recon = 0
        total_kl_div = 0
        y = to_gpu((torch.ones((x.shape[0], 10), dtype=torch.int64) * torch.arange(0,10)).T)
        for margin_y in y:
            # sample dependent on y
            mu, var = self.encoder(x, margin_y)
            z, kl_div = self.sample_z(mu, var)

            loss_recon = self.recon_loss(margin_y, x, z)
            total_loss_recon += y_dist[:,margin_y[0].item()] * loss_recon
            total_kl_div += y_dist[:,margin_y[0].item()] * kl_div
        return total_loss_recon.mean(), total_kl_div.mean(), y_entropy

    def _unsupervised_step(self, x):
        # unsupervised
        y_dist = torch.nn.functional.softmax(self.classifier(x), dim=-1)

        y_entropy = -(y_dist * torch.log(y_dist)).sum(dim=-1).mean()


        mu, var = self.encoder(x)
        total_loss_recon = 0
        y = to_gpu((torch.ones((mu.shape[0], 10), dtype=torch.int64) * torch.arange(0,10)).T)
        z, kl_div = self.sample_z(mu, var)
        for margin_y in y:

            loss_recon = self.recon_loss(margin_y, x, z)
            total_loss_recon += y_dist[:,margin_y[0].item()] * loss_recon

        return total_loss_recon.mean(), kl_div.mean(), y_entropy


    def unsupervised_step(self, x):
        if self.encoder.use_y:
            return self._unsupervised_step_y(x)
        else:
            return self._unsupervised_step(x)


    def sample_z(self, mu, var):

        p = normal.Normal(torch.zeros_like(mu).T, torch.ones_like(var).T)
        q = normal.Normal(mu.T, var.T)
        
        kl_div = kl.kl_divergence(q, p).sum(dim=0) # sum over dimensions

        # Reparametrized sample from q
        z = q.rsample().T
        return z, kl_div

    def recon_loss(self, y, x, z):
        x_hat = self.decoder(z, y)

        likelihood = binary_cross_entropy(x_hat, x.view(x.shape[0],-1), reduction="none")
        likelihood = likelihood.view(likelihood.size(0), -1).sum(1)

        return likelihood
