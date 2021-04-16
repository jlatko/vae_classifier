# Run options
from data.torchvision_dataset import get_datasets
import torch
import pyro
from modules.vae import VAE
from utils.experiment_utils import train, evaluate

USE_CUDA = torch.cuda.is_available()

# default configs
OPTIMIZER_CONFIG = {
    'lr': 1e-3,
}

VAE_CONFIG = {
    'z_dim': 4,
    'hidden_dim': 20
}


def run_training(optimizer_config=None, vae_config=None, num_epochs=100, test_frequency=5):
    if optimizer_config is None:
        optimizer_config = OPTIMIZER_CONFIG
    if vae_config is None:
        vae_config = VAE_CONFIG

    train_loader, test_loader = get_datasets(
        'MNIST', batch_size_train=256, batch_size_test=1024,
        download=True, # if already saved
        num_workers=2, pin_memory=USE_CUDA,
        transformation_kwargs={'flip': False, 'crop': False, 'normalize': None})

    # clear param store
    pyro.clear_param_store()

    # setup the VAE
    vae = VAE(use_cuda=USE_CUDA, **vae_config)

    # setup the optimizer
    # adam_args = {"lr": LEARNING_RATE}
    # optimizer = Adam(adam_args)
    optimizer = torch.optim.Adam(vae.parameters(), **optimizer_config)


    train_elbo = []
    test_elbo = []
    # training loop
    for epoch in range(num_epochs):
        total_epoch_loss_train = train(vae, train_loader, optimizer, use_cuda=USE_CUDA)
        train_elbo.append(-total_epoch_loss_train)

        if epoch % test_frequency == 0:
            print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

            # report test diagnostics
            total_epoch_loss_test, classification_accuracy = evaluate(vae, test_loader, use_cuda=USE_CUDA)
            test_elbo.append(-total_epoch_loss_test)
            print("[epoch %03d] average test loss: %.4f, accuracy %.4f" % (epoch, total_epoch_loss_test, classification_accuracy))

    return {
        'vae': vae,
    }