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
        self.hidden_dim = vae_config['hidden_dim']

        self.encoder = Encoder(self.z_dim, self.hidden_dim)
        self.classifier = Classifier(self.hidden_dim)
        self.decoder = Decoder(self.z_dim, self.hidden_dim)

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
        loss_u_recon, loss_u_KL = self.unsupervised_step(x_unsupervised)
        loss = loss_s_recon + loss_u_recon + self.kl_weight * (loss_s_KL + loss_u_KL) + self.a * loss_class
        
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
        loss_class, loss_s_recon, loss_s_KL, y_logits = self.supervised_step(x, y)
        loss_u_recon, loss_u_KL = self.unsupervised_step(x)
        loss = loss_s_recon 
        loss += loss_u_recon 
        loss += self.kl_weight * (loss_s_KL + loss_u_KL) 
        loss += self.a * loss_class

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
        mu, var = self.encoder(x)
        y_dist = self.classifier(x) # <- only for class loss)
        loss_class = cross_entropy(y_dist, y, reduction='mean')
        z, kl_div = self.sample_z(mu, var)
        loss_recon = self.recon_loss(y, x, z)
        return loss_class, loss_recon.mean(), kl_div.mean(), y_dist


    def unsupervised_step(self, x):
        mu, var = self.encoder(x)
        # unsupervised
        y_dist = torch.nn.functional.softmax(self.classifier(x), dim=-1)
        # for i in N_samples: # <- we can ignore this loop for now 
        if self.use_a: # also ignore this for now
            a = self.a.sample() # TODO implement true a
        total_loss_recon = 0
        y = to_gpu((torch.ones((mu.shape[0], 10), dtype=torch.int64) * torch.arange(0,10)).T)
        z, kl_div = self.sample_z(mu, var)
        for margin_y in y:
            loss_recon = self.recon_loss(margin_y, x, z)
            total_loss_recon += y_dist[:,margin_y[0].item()] * loss_recon

        return total_loss_recon.mean(), kl_div.mean()

    def sample_z(self, mu, var):

        p = normal.Normal(torch.zeros_like(mu), torch.ones_like(var))
        q = normal.Normal(mu, var)
        
        kl_div = kl.kl_divergence(q, p)

        # Reparametrized sample from q
        z = q.rsample()
        return z, kl_div

    def recon_loss(self, y, x, z):
        x_hat = self.decoder(z, y)

        likelihood = binary_cross_entropy(x_hat, x.view(x.shape[0],-1), reduction="none")
        likelihood = likelihood.view(likelihood.size(0), -1).sum(1)

        return likelihood

        ELBO = torch.mean(likelihood) - (kl_weight*torch.mean(kl_div))
        
        # minus sign as we want to maximise ELBO
        return -ELBO, kl_weight*kl_div.mean() 
