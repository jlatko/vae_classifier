import os

import torch
import wandb

from utils.torch_utils import to_gpu
import pyro.distributions as dist
import numpy as np
import matplotlib.pyplot as plt


def run_visualizations(model, test_dataloader, path='./fig/'):
    model.eval()
    model = to_gpu(model) # will this work?

    if not os.path.exists(path):
        os.makedirs(path)
    # import IPython
    # IPython.embed()
    filenames = test_reconstructions(model, test_dataloader, path=path)
    filenames += digit_reconstructions(model, test_dataloader, path=path)
    for filename in filenames:
        wandb.save(filename)
    # TODO: sync vis

def digit_reconstructions(model, test_dataloader, n=5, path='./fig/'):
    filenames = []

    x, y = next(iter(test_dataloader))
    x = to_gpu(x)
    y = to_gpu(y)

    x = x[:n]
    y = y[:n]
    z_loc, z_scale = model.vae.encoder(x)

    z_0 = torch.zeros_like(z_loc[:1])
    z_s = dist.Normal(torch.zeros_like(z_loc), torch.ones_like(z_scale)).sample()

    z = torch.cat([z_0, z_s, z_loc], dim=0)

    recons = []
    for i in range(10):
        recons.append(model.vae.decoder(z, torch.ones(1 + n*2, dtype=np.int) * i).reshape(-1, 28, 28).cpu().detach().numpy())

    plt.figure(figsize=(12, 12))

    for j in range(n):
        plt.subplot(2*n + 1, 11, (j + n + 1) * 11 + 1)
        plt.imshow(x[j].reshape(28, 28).cpu().detach().numpy())
        plt.axis('off')

    for j in range(n):
        plt.subplot(2*n + 1, 11, (j + 1) * 11 + 1)
        plt.text(0.5, 0.5, 'prior', size=16, ha='center', va='center')
        plt.axis('off')

    plt.subplot(2*n + 1, 11, 1)
    plt.text(0.5, 0.5, 'z=0', size=16, ha='center', va='center')
    plt.axis('off')

    for i in range(10):
        for j in range(2*n + 1):
            plt.subplot(2*n + 1, 11, j*11 + i + 2)
            plt.imshow(recons[i][j])
            plt.axis('off')

    filename = os.path.join(path, f'gen_digits.png')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(filename)
    filenames.append(filename)
    # TODO wandb plots
    return filenames


def test_reconstructions(model, test_dataloader, n=2, b=10, path='./fig/'):
    x, y = next(iter(test_dataloader))
    x = to_gpu(x)
    y = to_gpu(y)

    y_pred = model.vae.classifier(x)
    y_max = torch.argmax(y_pred, dim=-1)

    z_loc, z_scale = model.vae.encoder(x)
    # sample in latent space
    z_s = dist.Normal(z_loc, z_scale).sample()

    # decode the image (note we don't sample in image space)
    reconstructions_mean = model.vae.decoder(z_loc, y_max).reshape(-1, 28, 28).cpu().detach().numpy()
    reconstructions_sampled = model.vae.decoder(z_s, y_max).reshape(-1, 28, 28).cpu().detach().numpy()
    reconstructions_mean_sup = model.vae.decoder(z_loc, y).reshape(-1, 28, 28).cpu().detach().numpy()
    reconstructions_sampled_sup = model.vae.decoder(z_s, y).reshape(-1, 28, 28).cpu().detach().numpy()

    original = x.reshape(-1,28,28).cpu().detach().numpy()
    labels = y.cpu().detach().numpy()
    y_preds = y_max.cpu().detach().numpy()

    filenames = []
    for i in range(n):
        plot_reconstructions(original[i*b:(i+1)*b],
                             labels[i*b:(i+1)*b],
                             y_preds[i*b:(i+1)*b],
                             reconstructions_mean[i*b:(i+1)*b],
                             reconstructions_sampled[i*b:(i+1)*b],
                             reconstructions_mean_sup[i*b:(i+1)*b],
                             reconstructions_sampled_sup[i*b:(i+1)*b])
        filename = os.path.join(path, f'test_recon_{i}.png')
        plt.savefig(filename)
        filenames.append(filename)

    return filenames



def plot_reconstructions(original,
                         labels,
                         pred,
                         reconstructions_mean,
                         reconstructions_sampled,
                         reconstructions_mean_sup,
                         reconstructions_sampled_sup):
    # TODO: this code is ugly :(, maybe refactor...
    l = len(original)
    plt.figure(figsize=(7, l*2))
    for i in range(l):
        plt.subplot(l, 5, i * 5 + 1)
        plt.title(f'A. In ({labels[i]})')
        plt.imshow(original[i])
        plt.axis('off')

        plt.subplot(l, 5, i * 5 + 2)
        plt.title(f'B. sup.')
        plt.imshow(reconstructions_mean_sup[i])
        plt.axis('off')

        plt.subplot(l, 5, i * 5 + 3)
        plt.title(f'C. s sup.')
        plt.imshow(reconstructions_sampled_sup[i])
        plt.axis('off')

        plt.subplot(l, 5, i * 5 + 4)
        plt.title(f'D. ({pred[i]})')
        plt.imshow(reconstructions_mean[i])
        plt.axis('off')

        plt.subplot(l, 5, i * 5 + 5)
        plt.title(f'E. s ({pred[i]})')
        plt.imshow(reconstructions_sampled[i])
        plt.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.05)
