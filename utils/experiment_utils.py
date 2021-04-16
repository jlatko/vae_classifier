from sklearn.metrics import accuracy_score
import numpy as np
from criterion.get_losses import get_losses


def train(vae, train_loader, optimizer, use_cuda=False, a=1):
    # initialize loss accumulator
    epoch_loss = 0.
    classification_accuracy = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x, y in train_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        x_unsupervised = None  # TODO

        loss_unsup, loss_sup, loss_class, y_logits = get_losses(vae, y, x, x_unsupervised)
        loss = loss_unsup + loss_sup + a * loss_class
        epoch_loss += loss
        # step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train


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