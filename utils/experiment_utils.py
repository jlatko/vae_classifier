from sklearn.metrics import accuracy_score
import numpy as np
from criterion.get_losses import get_losses


def train(vae, train_loader_labelled, train_loader_missing, optimizer, use_cuda=False, a=1):
    # initialize loss accumulator
    epoch_loss = 0.
    epoch_loss_sup = 0.
    epoch_loss_unsup = 0.
    epoch_loss_class = 0.
    classification_accuracy = 0.

    n_sup = 0
    n_unsup = 0

    for (x, y), (x_unsupervised, _) in zip(train_loader_labelled, train_loader_missing):
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        # calculate loss
        loss_unsup, loss_sup, loss_class, y_logits = get_losses(vae, y, x, x_unsupervised)
        loss = loss_unsup + loss_sup + a * loss_class
        # step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # keep track of metrics
        epoch_loss_sup += loss_sup
        epoch_loss_unsup += loss_unsup
        epoch_loss_class += loss_class
        epoch_loss += loss
        n_sup += len(x)
        n_unsup += len(x_unsupervised) if x_unsupervised is not None else 1

    return {
        "loss": epoch_loss / (n_sup + n_unsup),
        "loss_supervised": epoch_loss_sup / n_sup,
        "loss_unsupervised": epoch_loss_unsup / n_unsup,
        "loss_class": epoch_loss_class / n_sup,
    }


def evaluate(vae, test_loader, use_cuda=False, a=1):
    # initialize loss accumulator
    test_loss = 0.
    accuracies = []
    # compute the loss over the entire test set
    for x, y in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        x_unsupervised = None  # TODO

        loss_unsup, loss_sup, loss_class, y_logits = get_losses(vae, y, x, x_unsupervised)
        loss = loss_unsup + loss_sup + a * loss_class
        test_loss += loss
        accuracies.append(accuracy_score(y.cpu().numpy(), np.argmax(y_logits.detach().cpu().numpy(), axis=1)))

    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test

    return total_epoch_loss_test, np.mean(accuracies)