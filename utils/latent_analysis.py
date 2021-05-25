import os

import wandb
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.torch_utils import to_gpu
import numpy as np
from sklearn.model_selection import cross_val_score
import itertools
import pandas as pd
import matplotlib.pyplot as plt


def numpy_save_synch(path, arr, filename):
    fname = os.path.join(path, filename)
    np.save(fname, arr)
    wandb.save(fname)

def analyze_latents(model, test_dataloader, path='./fig/'):
    if not os.path.exists(path):
        os.makedirs(path)

    # get z, y
    z_loc, z_scale, y = get_latents(model, test_dataloader)

    # save
    numpy_save_synch(path, z_loc, 'z_loc.npy')
    numpy_save_synch(path, z_scale, 'z_scale.npy')
    numpy_save_synch(path, y, 'y.npy')

    # try classifying y from z_loc
    acc = cv_classify(z_loc, y)

    print("Z -> y classification accuracy: ", acc)
    wandb.log({"z_acc": acc})

    # simple stats
    print("z_loc variance", np.std(z_loc))
    print("z_scale mean", np.mean(z_scale))
    print("z_scale variance", np.std(z_scale))

    for i in range(z_loc.shape[-1]):
        wandb.log({"z_loc": wandb.Histogram(z_loc[:, i]), "step": i})
        wandb.log({"z_scale": wandb.Histogram(z_scale[:, i]), "step": i})


        wandb.log({"z_loc_std": np.std(z_loc[:, i]), "step": i})
        wandb.log({"z_scale_mean": np.mean(z_scale[:, i]), "step": i})
        wandb.log({"z_scale_std": np.std(z_scale[:, i]), "step": i})

    # pca
    n_components = min(3, z_loc.shape[-1])
    pca = PCA(n_components=n_components)
    z_pca = pca.fit_transform(z_loc)
    numpy_save_synch(path, z_pca, 'z_pca.npy')

    for i, j in itertools.combinations(list(range(n_components)), 2):
        plot_scatter(z_pca[:,i], z_pca[:,j], y, path, x1_name=f"PCA_{i}", x2_name=f"PCA_{j}")

    z_tsne = TSNE(n_components=2).fit_transform(z_loc)
    numpy_save_synch(path, z_tsne, 'z_tsne.npy')
    plot_scatter(z_tsne[:, 0], z_tsne[:, 1], y, path, x1_name="TSNE_0", x2_name="TSNE_1")

    df = pd.DataFrame(z_pca, columns=[f"PCA_{i}" for i in range(n_components)])
    df_z = pd.DataFrame(z_loc, columns=[f"z_{i}" for i in range(z_loc.shape[-1])])
    df_tsne = pd.DataFrame(z_tsne, columns=["TSNE_0", "TSNE_1"])
    df = pd.concat([df, df_z, df_tsne], axis=1)
    df["y"] = y
    table = wandb.Table(dataframe=df)
    wandb.log({f"latent_z": wandb.plot.scatter(table, f"TSNE_0", f"TSNE_1")})


def plot_scatter(x1, x2, y, path, x1_name=None, x2_name=None):
    plt.figure(figsize=(10,10))
    for t in range(10):
        mask = y == t
        plt.scatter(x1[mask], x2[mask], label=str(t), alpha=0.2)
    plt.legend()
    plt.xlabel(x1_name)
    plt.ylabel(x2_name)

    plt.savefig(os.path.join(path, f'{x1_name}_{x2_name}.png'))
    wandb.log({f'{x1_name}_{x2_name}': wandb.Image(plt)})


def cv_classify(x, y):
    clf = svm.SVC(kernel='linear', C=1, random_state=42)
    scores = cross_val_score(clf, x, y, cv=5, scoring='accuracy')
    return np.mean(scores)


def get_latents(model, dataloader):
    z_locs = []
    z_scales = []
    ys = []
    for x, y in dataloader:
        x = to_gpu(x)
        y = to_gpu(y)
        if model.encoder.use_y:
            z_loc, z_scale = model.encoder(to_gpu(x), y)
        else:
            z_loc, z_scale = model.encoder(to_gpu(x))
        ys.append(y.detach().cpu().numpy())
        z_locs.append(z_loc.detach().cpu().numpy())
        z_scales.append(z_scale.detach().cpu().numpy())
    return np.concatenate(z_locs, axis=0), np.concatenate(z_scales, axis=0), np.concatenate(ys, axis=0)