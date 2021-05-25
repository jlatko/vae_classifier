import torch

from utils.model_utils import build_dense_nn


class Encoder(torch.nn.Module):
    def __init__(self, z_dim, shared_units, separate_units, use_batch_norm=False, use_y=False, y_embedding_size=0):
        super().__init__()
        self.use_y = use_y
        if use_y:
            assert y_embedding_size > 0
            print("using y")
            self.y_net = torch.nn.Embedding(10, y_embedding_size)
        else:
            y_embedding_size = 0

        # setup the three linear transformations used
        self.fc1 = build_dense_nn(784, shared_units,
                                  use_batch_norm=use_batch_norm,
                                  last_batchnorm=True,
                                  last_activation='relu')

        self.fc21 = build_dense_nn(shared_units[-1] + y_embedding_size, separate_units + [z_dim],
                                  use_batch_norm=use_batch_norm,
                                  last_batchnorm=False,
                                  last_activation=None)
        self.fc22 = build_dense_nn(shared_units[-1] + y_embedding_size, separate_units + [z_dim],
                                  use_batch_norm=use_batch_norm,
                                  last_batchnorm=False,
                                  last_activation=None)

    def forward(self, x, y=None):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, 784)
        # then compute the hidden units
        hidden = self.fc1(x)
        if self.use_y:
            hidden = torch.cat([hidden, self.y_net(y)], dim=-1)
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale