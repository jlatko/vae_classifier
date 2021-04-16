import torch

class Decoder(torch.nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the two linear transformations used
        self.z_net = torch.nn.Sequential(
            torch.nn.Linear(z_dim, hidden_dim),
            torch.nn.Softplus()
        )
        self.y_net = torch.nn.Sequential(
            torch.nn.Embedding(10, hidden_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.Softplus(),
            torch.nn.Linear(hidden_dim, 784),
            torch.nn.Sigmoid()
        )

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

