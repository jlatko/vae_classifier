import torch
from torchvision import datasets, transforms, models
from collections.abc import Iterable

# this shit seems to be required to download
from six.moves import urllib

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


def get_datasets(name, batch_size_train=256, batch_size_test=1024,
                 num_workers=2, pin_memory=True, transformation_kwargs=None):
    # TODO: validation
    dataset = getattr(datasets, name)

    if transformation_kwargs is None:
        transformation_kwargs = {}
    train_transform, test_transform = get_transformations(**transformation_kwargs)

    train_dataset = dataset(f'{name.lower()}_data', train=True, download=True,
                transform=train_transform)

    test_dataset = dataset(f'{name.lower()}_data', train=False, download=False,
                transform=test_transform)

    return train_dataset, test_dataset


def get_transformations(flip=False, crop=False, crop_size=32, crop_padding=4, normalize=None):
    train_transformations = []
    test_transformations = []

    if flip:
        train_transformations.append(transforms.RandomHorizontalFlip())

    if crop:
        train_transformations.append(transforms.RandomCrop(crop_size, padding=crop_padding))

    # to tensor
    train_transformations.append(transforms.ToTensor())
    test_transformations.append(transforms.ToTensor())

    if normalize == 'cifar':
        train_transformations.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        test_transformations.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    elif isinstance(normalize, Iterable):
        train_transformations.append(transforms.Normalize(normalize[0], normalize[1]))
        test_transformations.append(transforms.Normalize(normalize[0], normalize[1]))

    train_transform = transforms.Compose(train_transformations)
    test_transform = transforms.Compose(test_transformations)

    return train_transform, test_transform