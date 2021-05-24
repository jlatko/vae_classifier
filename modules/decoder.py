import torch

from utils.model_utils import build_dense_nn


class Decoder(torch.nn.Module):
    def __init__(self, z_dim, z_units, y_embedding_size, decoder_units, use_batch_norm=False):
        super().__init__()
        self.z_net = build_dense_nn(z_dim, z_units,
                         use_batch_norm=use_batch_norm,
                         last_batchnorm=True,
                         last_activation='relu')

        self.y_net = torch.nn.Embedding(10, y_embedding_size)

        self.decoder = build_dense_nn(z_units[-1] + y_embedding_size, decoder_units + [784],
                         use_batch_norm=use_batch_norm,
                         last_batchnorm=False,
                         last_activation='sigmoid')

    def forward(self, z, y):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden_z = self.z_net(z)
        # embed y
        hidden_y = self.y_net(y)
        # concatenate
        hidden = torch.cat([hidden_z, hidden_y], dim=-1)
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_img = self.decoder(hidden)
        return loc_img

