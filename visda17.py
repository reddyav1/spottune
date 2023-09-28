import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader
from random import sample
import math

from pathlib import Path

means =  [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    # TODO: experiment with transforms
                transforms.Resize(72),
                transforms.CenterCrop(72),
                transforms.ToTensor(),
                transforms.Normalize(means, stds)])

test_transform = transforms.Compose([
    # TODO: experiment with transforms
                transforms.Resize(72),
                transforms.CenterCrop(72),
                transforms.ToTensor(),
                transforms.Normalize(means, stds)])

def downsample_dataset(dataset, fraction=1.0):
    n = len(dataset)
    indices = sample(range(n), math.floor(n*fraction))
    
    dataset.samples = [dataset.samples[idx] for idx in indices]
    dataset.targets = [s[1] for s in dataset.samples]


def get_visda_dataloaders(train_dir,
                          val_dir,
                          test_dir,
                          batch_size_train=128,
                          batch_size_test=128,
                          train_fraction=1.0,
                          val_fraction=1.0,
                          test_fraction=1.0,
                          train_transform=train_transform,
                          test_transform=test_transform
):
    trainset = ImageFolder(train_dir, transform=train_transform)
    valset = ImageFolder(val_dir, transform=test_transform)
    testset = ImageList(test_dir,
                        list(trainset.class_to_idx.keys()),
                        Path(test_dir) / 'image_list_gt.txt',
                        transform=test_transform,
                        return_path=False
    )

    print(trainset.class_to_idx)
    print(valset.class_to_idx)
    print(testset.class_to_idx)

    # downsample datasets
    downsample_dataset(trainset, train_fraction)
    downsample_dataset(valset, val_fraction)
    downsample_dataset(testset, test_fraction)

    trainloader = DataLoader(trainset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=8)
    valloader = DataLoader(valset, batch_size=batch_size_test, shuffle=True, pin_memory=True, num_workers=8)
    testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=True, pin_memory=True, num_workers=8)

    return (trainloader, valloader, testloader)

import os
from typing import Optional, Callable, Tuple, Any, List
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader


class ImageList(datasets.VisionDataset):
    """A generic Dataset class for image classification
    Args:
        root (str): Root directory of dataset
        classes (list[str]): The names of all the classes
        data_list_file (str): File to read the image list from.
        transform (callable, optional): A function/transform that  takes in an PIL image \
            and returns a transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    .. note:: In `data_list_file`, each line has 2 values in the following format.
        ::
            source_dir/dog_xxx.png 0
            source_dir/cat_123.png 1
            target_dir/dog_xxy.png 0
            target_dir/cat_nsdf3.png 1
        The first value is the relative path of an image, and the second value is the label of the corresponding image.
        If your data_list_file has different formats, please over-ride :meth:`~ImageList.parse_data_file`.
    """

    def __init__(self, root: str, classes: List[str], data_list_file: str,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 return_path = False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = self.parse_data_file(data_list_file)
        self.classes = classes
        self.class_to_idx = {cls: idx
                             for idx, cls in enumerate(self.classes)}
        self.loader = default_loader
        self.data_list_file = data_list_file
        self.return_path = return_path

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index
            return (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)

        if self.return_path:
            return img, target, path
        else:
            return img, target

    def __len__(self) -> int:
        return len(self.samples)

    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        """Parse file to data list
        Args:
            file_name (str): The path of data file
            return (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                split_line = line.split()
                target = split_line[-1]
                path = ' '.join(split_line[:-1])
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = int(target)
                data_list.append((path, target))
        return data_list

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)

    @classmethod
    def domains(cls):
        """All possible domain in this dataset"""
        raise NotImplemented