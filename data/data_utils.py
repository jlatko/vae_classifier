import numpy as np
from torch.utils.data import DataLoader, Subset

def get_missing_labels_dataset(train_dataset, labelled_per_class=600, seed=0):
    np.random.seed(seed)

    targets = np.array(train_dataset.targets)
    labelled_idx = np.array([], dtype=np.int64)
    for cls in range(10):
        cls_indices = np.argwhere(targets == cls)[:, 0]
        labelled_idx = np.append(labelled_idx,
                                 np.random.choice(cls_indices,
                                                  size=labelled_per_class,
                                                  replace=False))

    unlabelled_idx = np.array(list(set(list(range(len(targets)))) - set(labelled_idx)))
    data_labelled = Subset(train_dataset, labelled_idx)
    data_missing = Subset(train_dataset, unlabelled_idx)

    print("Splitting datasets")
    print("labelled: ", len(labelled_idx))
    print("unlabelled: ", len(unlabelled_idx))
    print("fraction: ", len(labelled_idx)/len(targets))
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