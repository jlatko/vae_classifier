import pytorch_lightning as pl
import torch
from torch.utils.data import RandomSampler

from data.data_utils import get_missing_labels_dataset, DummyIterator

from data.torchvision_dataset import get_datasets


class SemiSupervised(pl.LightningDataModule):
    def __init__(self, fraction_missing=0.5, missing_batch_size=128, labelled_batch_size=128, n_steps=200):
        super(SemiSupervised, self).__init__()
        self.fraction_missing = fraction_missing
        self.missing_batch_size = missing_batch_size
        self.labelled_batch_size = labelled_batch_size
        self.n_steps = n_steps


    def setup(self, stage=None):
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """
        self.train_dataset, self.test_dataset = get_datasets(
            'MNIST',
            transformation_kwargs={'flip': False, 'crop': False, 'normalize': None})

    def train_dataloader(self):
        print("setting up dataloaders")
        if self.fraction_missing == 0:
            train_loader_labelled = torch.utils.data.DataLoader(self.train_dataset,
                                                                batch_size=self.labelled_batch_size, shuffle=True,
                                                                num_workers=2)
            train_loader_missing = DummyIterator(len(train_loader_labelled))
        else:
            data_labelled, data_missing = get_missing_labels_dataset(self.train_dataset, fraction_missing=self.fraction_missing)

            # use explicit sampler to specify number of samples per epoch
            # labelled samples
            sampler = RandomSampler(data_labelled, num_samples=self.labelled_batch_size * self.n_steps, replacement=True)
            train_loader_labelled = torch.utils.data.DataLoader(data_labelled,
                                                                batch_size=self.labelled_batch_size, sampler=sampler,
                                                                num_workers=2)
            # unlabelled samples
            sampler = RandomSampler(data_missing, num_samples=self.missing_batch_size * self.n_steps, replacement=True)
            train_loader_missing = torch.utils.data.DataLoader(data_missing,
                                                               batch_size=self.labelled_batch_size, sampler=sampler,
                                                               num_workers=2)

            # use both together
            return {'labelled': train_loader_labelled, 'missing': train_loader_missing}

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=1024, shuffle=False, num_workers=2)

    def test_dataloader(self):
        return self.val_dataloader() # TODO?