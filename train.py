# Run options
from torch.utils.data import RandomSampler

from data.data_utils import get_missing_labels_dataset, DummyIterator
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

assert pyro.__version__.startswith('1.6.0')
# without that Bernoulli raises: The value argument must be within the support
# TODO: maybe we should keep it...
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)

def setup_data(fraction_missing=0.5, missing_batch_size=128, labelled_batch_size=128, n_steps=200):

    print('Preparing data')
    # get datasets from torchvision
    train_dataset, test_dataset = get_datasets(
        'MNIST',
        transformation_kwargs={'flip': False, 'crop': False, 'normalize': None})

    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=1024, shuffle=False, num_workers=2, pin_memory=USE_CUDA)

    # get dataloaders
    if fraction_missing == 0:
        train_loader_labelled = torch.utils.data.DataLoader(train_dataset,
            batch_size=labelled_batch_size, shuffle=True, num_workers=2, pin_memory=USE_CUDA)
        train_loader_missing = DummyIterator(len(train_loader_labelled))
    else:
        data_labelled, data_missing = get_missing_labels_dataset(train_dataset, fraction_missing=fraction_missing)

        # use explicit sampler to specify number of samples per epoch
        # labelled samples
        sampler = RandomSampler(data_labelled, num_samples=labelled_batch_size * n_steps, replacement=True)
        train_loader_labelled = torch.utils.data.DataLoader(data_labelled,
            batch_size=labelled_batch_size, sampler=sampler, num_workers=2, pin_memory=USE_CUDA)
        # unlabelled samples
        sampler = RandomSampler(data_missing, num_samples=missing_batch_size * n_steps, replacement=True)
        train_loader_missing = torch.utils.data.DataLoader(data_missing,
            batch_size=labelled_batch_size, sampler=sampler, num_workers=2, pin_memory=USE_CUDA)

    return train_loader_labelled, train_loader_missing, test_loader

def setup_model(optimizer_config=None, vae_config=None):
    if optimizer_config is None:
        optimizer_config = OPTIMIZER_CONFIG
    if vae_config is None:
        vae_config = VAE_CONFIG

    # clear param store
    pyro.clear_param_store()

    print("Prepare the model")
    # setup the VAE
    vae = VAE(use_cuda=USE_CUDA, **vae_config)

    # setup the optimizer
    optimizer = torch.optim.Adam(vae.parameters(), **optimizer_config)

    return vae, optimizer

def run_training(optimizer_config=None, vae_config=None, num_epochs=100, test_frequency=1, a=1000, fraction_missing=0.99, missing_batch_size=128, labelled_batch_size=128, n_steps=200):
    train_loader_labelled, train_loader_missing, test_loader = setup_data(fraction_missing, missing_batch_size, labelled_batch_size, n_steps)
    vae, optimizer = setup_model(optimizer_config, vae_config)

    results_train = []
    results_test = []

    # training loop
    for epoch in range(num_epochs):
        results = train(vae, train_loader_labelled, train_loader_missing, optimizer, use_cuda=USE_CUDA, a=a)
        results_train.append(results)
        print(f"[epoch {epoch}] average training loss: {results['loss']:.4f} "
              f"| sup: {results['loss_supervised']:.4f} "
              f"| unsup: {results['loss_unsupervised']:.4f}"
              f"| class: {results['loss_class']:.4f}"
              f"| acc: {results['accuracy']:.4f}")

        if epoch % test_frequency == 0:

            # report test diagnostics
            results = evaluate(vae, test_loader, use_cuda=USE_CUDA, a=a)
            results_test.append(results)
            print(f"TEST [e {epoch}]: average loss: {results['loss']:.4f} "
                  f"| sup: {results['loss_supervised']:.4f} "
                  f"| unsup: {results['loss_unsupervised']:.4f}"
                  f"| class: {results['loss_class']:.4f}"
                    f"| acc: {results['accuracy']:.4f}")

    return {
        'vae': vae,
        'results_train': results_train,
        'results_test': results_test
    }

if __name__ == '__main__':
    run_training()