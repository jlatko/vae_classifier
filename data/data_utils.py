import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

def get_missing_labels_dataset(train_dataset, fraction_missing=0.5, seed=0):
    targets = np.array(train_dataset.targets)
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=fraction_missing,
        shuffle=True,
        stratify=targets,
        random_state=seed)  # seed is important
    data_labelled = Subset(train_dataset, train_idx)
    data_missing = Subset(train_dataset, val_idx)
    return data_labelled, data_missing

class DummyIterator:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return self

    def __len__(self):
        return self.n

    def __next__(self):
        return None, None