# Run options
import torch
import pyro
from modules.classifier import Classifier
from train import setup_data
from utils.experiment_utils import train_classifier, evaluate_classifier

USE_CUDA = torch.cuda.is_available()

# default configs
OPTIMIZER_CONFIG = {
    'lr': 1e-3,
}

MODEL_CONFIG = {
    'hidden_dim': 20
}

assert pyro.__version__.startswith('1.6.0')
# without that Bernoulli raises: The value argument must be within the support
# TODO: maybe we should keep it...
pyro.distributions.enable_validation(False)
pyro.set_rng_seed(0)

def setup_model(optimizer_config=None, model_config=None):
    if optimizer_config is None:
        optimizer_config = OPTIMIZER_CONFIG
    if model_config is None:
        model_config = MODEL_CONFIG

    # clear param store
    pyro.clear_param_store()

    print("Prepare the model")
    # setup the VAE
    model = Classifier(**model_config)
    if USE_CUDA:
        model.cuda()
    # setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_config)

    return model, optimizer

def run_training(optimizer_config=None, model_config=None, num_epochs=100, test_frequency=1, fraction_missing=0.99, missing_batch_size=128, labelled_batch_size=128, n_steps=200):
    train_loader_labelled, train_loader_missing, test_loader = setup_data(fraction_missing, missing_batch_size, labelled_batch_size, n_steps)
    model, optimizer = setup_model(optimizer_config, model_config)

    results_train = []
    results_test = []

    # training loop
    for epoch in range(num_epochs):
        results = train_classifier(model, train_loader_labelled, optimizer, use_cuda=USE_CUDA)
        results_train.append(results)
        print(f"[epoch {epoch}] average training loss: {results['loss']:.4f} "
              f"| acc: {results['accuracy']:.4f}")

        if epoch % test_frequency == 0:

            # report test diagnostics
            results = evaluate_classifier(model, test_loader, use_cuda=USE_CUDA)
            results_test.append(results)
            print(f"TEST [e {epoch}]: average loss: {results['loss']:.4f} "
              f"| acc: {results['accuracy']:.4f}")

    return {
        'model': model,
        'results_train': results_train,
        'results_test': results_test
    }

if __name__ == '__main__':
    run_training()