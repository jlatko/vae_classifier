import torch

import pytorch_lightning as pl
from torch.nn.functional import cross_entropy
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
        self.hidden_dim = vae_config['hidden_dim']

        self.encoder = Encoder(self.z_dim, self.hidden_dim)
        self.classifier = Classifier(self.hidden_dim)
        self.decoder = Decoder(self.z_dim, self.hidden_dim)

        if self.use_cuda:
            self.cuda()

        self.a = a
        self.use_a = False
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.optimizer_config = optimizer_config


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.optimizer_config)


    def training_step(self, batch, batch_idx): # pylint: disable=unused-argument
        x, y = batch
        x_unsupervised, _ = batch["missing"]
        loss_class, loss_s_recon, loss_s_KL = self.supervised_step(x, y)
        loss_u_recon, loss_u_KL, y_logits = self.unsupervised_step(x_unsupervised)
        loss = loss_s_recon + loss_u_recon + loss_s_KL + loss_u_KL + self.a * loss_class
        
        probs = torch.nn.functional.softmax(y_logits, dim=1)

        self.train_acc(probs, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_sup", loss_s_recon, on_step=False, on_epoch=True)
        self.log("train_loss_sup_KL", loss_s_KL, on_step=False, on_epoch=True)
        self.log("train_loss_unsup", loss_u_recon, on_step=False, on_epoch=True)
        self.log("train_loss_unsup_KL", loss_u_KL, on_step=False, on_epoch=True)
        self.log("train_loss_class", loss_class, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        loss_class, loss_s_recon, loss_s_KL = self.supervised_step(x, y)
        loss_u_recon, loss_u_KL, y_logits = self.unsupervised_step(x)
        loss = loss_s_recon + loss_u_recon + loss_s_KL + loss_u_KL + self.a * loss_class

        probs = torch.nn.functional.softmax(y_logits, dim=1)

        self.val_acc(probs, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_loss_sup", loss_s_recon, on_step=False, on_epoch=True)
        self.log("val_loss_sup_KL", loss_s_KL, on_step=False, on_epoch=True)
        self.log("val_loss_unsup", loss_u_recon, on_step=False, on_epoch=True)
        self.log("val_loss_unsup_KL", loss_u_KL, on_step=False, on_epoch=True)
        self.log("val_loss_class", loss_class, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)


    def supervised_step(self, x, y):
        mu, log_var = self.encoder(x)
        y_dist = self.classifier(x) # <- only for class loss)
        loss_class = cross_entropy(y_dist, y, reduction='none')
        loss_recon, loss_KL = self.ELBO_loss(y, x, mu, log_var)
        return loss_class, loss_recon, loss_KL


    def unsupervised_step(self, x):
        mu, log_var = self.encoder(x)
        # unsupervised
        y_dist = self.classifier(x)
        # for i in N_samples: # <- we can ignore this loop for now 
        if self.use_a: # also ignore this for now
            a = self.a.sample() # TODO implement true a
        total_loss_recon = 0
        y = torch.arange(0,10)
        loss_recon, loss_KL = self.ELBO_loss(y, x, mu, log_var)
        total_loss_recon = y_dist * loss_recon

        return total_loss_recon, loss_KL, y_dist


    def ELBO_loss(self, y, x, mu, log_var, kl_weight=1):
        # Sum over features
        
        sigma = torch.exp(log_var*2)

        p = normal.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        q = normal.Normal(mu, sigma)
        
        kl_div = kl.kl_divergence(q, p)

        z = q.rsample()

        x_hat = self.decoder(y, z)

        likelihood = -cross_entropy(x_hat, x, reduction="none")
        likelihood = likelihood.view(likelihood.size(0), -1).sum(1)


        ELBO = torch.mean(likelihood) - (kl_weight*torch.mean(kl_div))
        
        # minus sign as we want to maximise ELBO
        return -ELBO, kl_weight*kl_div.mean() 
