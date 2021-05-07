from sklearn.metrics import accuracy_score
import numpy as np
from criterion.get_losses import get_losses
import torch

cross_entropy = torch.nn.CrossEntropyLoss()


def train(vae, train_loader_labelled, train_loader_missing, optimizer, use_cuda=False, a=1):
    # initialize loss accumulator
    epoch_loss = 0.
    epoch_loss_sup = 0.
    epoch_loss_unsup = 0.
    epoch_loss_class = 0.
    accuracies = []

    n_sup = 0
    n_unsup = 0

    for (x, y), (x_unsupervised, _) in zip(train_loader_labelled, train_loader_missing):
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
            if x_unsupervised is not None:
                x_unsupervised = x_unsupervised.cuda()
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
        accuracies.append(accuracy_score(y.cpu().numpy(), np.argmax(y_logits.detach().cpu().numpy(), axis=1)))

    return {
        "loss": epoch_loss / (n_sup + n_unsup),
        "loss_supervised": epoch_loss_sup / n_sup,
        "loss_unsupervised": epoch_loss_unsup / n_unsup,
        "loss_class": epoch_loss_class / n_sup,
        "accuracy": np.mean(accuracies)
    }

def evaluate(vae, test_loader, use_cuda=False, a=1):# initialize loss accumulator
    epoch_loss = 0.
    epoch_loss_sup = 0.
    epoch_loss_unsup = 0.
    epoch_loss_class = 0
    accuracies = []

    n_sup = 0
    n_unsup = 0

    for x, y in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        # calculate loss

        _, loss_sup, loss_class, y_logits = get_losses(vae, y, x, None)
        loss_unsup, _, _, _ = get_losses(vae, y, None, x)
        loss = loss_unsup + loss_sup + a * loss_class

        # keep track of metrics
        epoch_loss_sup += loss_sup
        epoch_loss_unsup += loss_unsup
        epoch_loss_class += loss_class
        epoch_loss += loss
        n_sup += len(x)
        n_unsup += len(x)
        accuracies.append(accuracy_score(y.cpu().numpy(), np.argmax(y_logits.detach().cpu().numpy(), axis=1)))

    return {
        "loss": epoch_loss / (n_sup + n_unsup),
        "loss_supervised": epoch_loss_sup / n_sup,
        "loss_unsupervised": epoch_loss_unsup / n_unsup,
        "loss_class": epoch_loss_class / n_sup,
        "accuracy": np.mean(accuracies)
    }

def train_classifier(classifier, train_loader, optimizer, use_cuda=False):
    # initialize loss accumulator
    epoch_loss = 0.
    accuracies = []
    n = 0

    for x, y in train_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            y = y.cuda()
        # calculate loss
        y_logits = classifier(x)
        loss = cross_entropy(y_logits, y)
        # step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        accuracies.append(accuracy_score(y.cpu().numpy(), np.argmax(y_logits.detach().cpu().numpy(), axis=1)))

        # keep track of metrics
        epoch_loss += loss
        n += len(x)

    return {
        "loss": epoch_loss / n,
        "accuracy": np.mean(accuracies)
    }


def evaluate_classifier(classifier, test_loader, use_cuda=False, a=1):
    # initialize loss accumulator
    test_loss = 0.
    accuracies = []
    # compute the loss over the entire test set
    for x, y in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        y_logits = classifier(x)
        loss = cross_entropy(y_logits, y)
        test_loss += loss
        accuracies.append(accuracy_score(y.cpu().numpy(), np.argmax(y_logits.detach().cpu().numpy(), axis=1)))

    n = len(test_loader.dataset)

    return {
        "loss": test_loss / n,
        "accuracy": np.mean(accuracies)
    }