import pyro
import torch
import pyro.distributions as dist
import pyro.contrib.examples.util

from modules.classifier import Classifier
from modules.decoder import Decoder
from modules.encoder import Encoder
from utils.model_utils import build_dense_nn
from utils.torch_utils import to_gpu


class VAE(torch.nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, encoder_config, decoder_config, classifier_units, use_batch_norm=False, z_dim=50, use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim,
                               encoder_config['shared_units'],
                               encoder_config['separate_units'],
                               use_batch_norm=use_batch_norm)
        self.classifier = Classifier(classifier_units, use_batch_norm=use_batch_norm)
        self.decoder = Decoder(z_dim, decoder_config['z_units'], decoder_config['y_embedding_size'],
                               decoder_config['decoder_units'], use_batch_norm=use_batch_norm)

        if use_cuda:
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    def model_supervised(self, x, y):
        """ p(x | y, z), where y is observed
        """
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            loc_img = self.decoder(z, y)
            # TODO: Bernoulli now raises: The value argument must be within the support
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))
            # pyro.sample("obs", dist.Normal(loc_img, 0.1).to_event(1), obs=x.reshape(-1, 784))

    def model_unsupervised(self, x):
        """ p(x|y,z), where y ~ q(y|x) (unobserved)
        """
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
            y = pyro.sample("y", dist.Categorical(to_gpu(torch.ones(10))))
            loc_img = self.decoder(z, y)

            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))
            # pyro.sample("obs", dist.Normal(loc_img, 0.1).to_event(1), obs=x.reshape(-1, 784))

    def guide_supervised(self, x, y):
        """ q(z|x)
        """
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            z_loc, z_scale = self.encoder(x)
            pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

    def guide_unsupervised(self, x):
        """ q(z|x) and q(y|x)
        """
        pyro.module("encoder", self.encoder)
        pyro.module("classifier", self.classifier)
        with pyro.plate("data", x.shape[0]):
            y_logits = self.classifier(x)
            y = pyro.sample("y", dist.Categorical(logits=y_logits))

            z_loc, z_scale = self.encoder(x)
            # sample the latent code z
            pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        y_logits = self.classifier(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        y = dist.Categorical(torch.nn.functional.softmax(y_logits, dim=-1)).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z, y)
        return loc_img
