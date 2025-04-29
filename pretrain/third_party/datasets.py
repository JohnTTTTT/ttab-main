# A collection of datasets class used in pretraining models.

# imports.
import os
import functools
import torch
import sys
sys.path.append("..")

from ttab.loads.datasets.dataset_shifts import NoShiftedData, SyntheticShiftedData
from ttab.loads.datasets.mnist import ColoredSyntheticShift
from ttab.loads.datasets.loaders import BaseLoader
from ttab.loads.datasets.datasets import OfficeHomeDataset, PACSDataset, CIFARDataset, WBirdsDataset, ColoredMNIST, AffectNetDataset

def get_train_dataset(config) -> BaseLoader:
    """Get the training dataset from `config`."""
    data_shift_class = functools.partial(NoShiftedData, data_name=config.data_name)
    if "cifar" in config.data_name:
        train_dataset = CIFARDataset(
            root=os.path.join(config.data_path, config.data_name),
            data_name=config.data_name,
            split="train",
            device=config.device,
            data_augment=True,
            data_shift_class=data_shift_class,
        )
        val_dataset = CIFARDataset(
            root=os.path.join(config.data_path, config.data_name),
            data_name=config.data_name,
            split="test",
            device=config.device,
            data_augment=False,
            data_shift_class=data_shift_class,
        )
    elif "officehome" in config.data_name:
        _data_names = config.data_name.split("_")
        dataset = OfficeHomeDataset(
            root=os.path.join(config.data_path, _data_names[0], _data_names[1]),
            device=config.device,
            data_augment=True,
            data_shift_class=data_shift_class,
        ).split_data(fractions=[0.9, 0.1], augment=[True, False], seed=config.seed)
        train_dataset, val_dataset = dataset[0], dataset[1]
    elif "pacs" in config.data_name:
        _data_names = config.data_name.split("_")
        dataset = PACSDataset(
            root=os.path.join(config.data_path, _data_names[0], _data_names[1]),
            device=config.device,
            data_augment=True,
            data_shift_class=data_shift_class,
        ).split_data(fractions=[0.9, 0.1], augment=[True, False], seed=config.seed)
        train_dataset, val_dataset = dataset[0], dataset[1]
    elif config.data_name == "waterbirds":
        train_dataset = WBirdsDataset(
            root=os.path.join(config.data_path, config.data_name),
            split="train",
            device=config.device,
            data_augment=True,
            data_shift_class=data_shift_class,
        )
        val_dataset = WBirdsDataset(
            root=os.path.join(config.data_path, config.data_name),
            split="val",
            device=config.device,
            data_augment=False,
        )
    elif config.data_name == "coloredmnist":
        data_shift_class = functools.partial(
            SyntheticShiftedData,
            data_name=config.data_name,
            seed=config.seed,
            synthetic_class=ColoredSyntheticShift(
                data_name=config.data_name, seed=config.seed
            ),
            version="stochastic",
        )
        train_dataset = ColoredMNIST(
            root=os.path.join(config.data_path, "mnist"),
            data_name=config.data_name,
            split="train",
            device=config.device,
            data_shift_class=data_shift_class,
        )
        val_dataset = ColoredMNIST(
            root=os.path.join(config.data_path, "mnist"),
            data_name=config.data_name,
            split="val",
            device=config.device,
            data_shift_class=data_shift_class,
        )
    elif config.data_name == "affectnet":
        from ttab.loads.datasets.datasets import ImageFolderDataset
        from ttab.api import PyTorchDataset, Batch
        # Import the new transforms from affectnet_transforms.py
        from ttab.loads.datasets.affectnet.affectnet_transforms import get_transform_affectnet
        import os

        input_size = 224
        # Get the new training and validation transforms.
        train_transform, val_transform = get_transform_affectnet(input_size=input_size)

        # Set the root paths (update if necessary).
        train_root = os.path.join(
            config.data_path,
            "/home/johnt/scratch/AffectNet7_37k_balanced/train",
        )
        val_root = os.path.join(
            config.data_path,
            "/home/johnt/scratch/AffectNet7_37k_balanced/val",
        )

        # Create the underlying datasets.
        train_dataset = ImageFolderDataset(root=train_root, transform=train_transform)
        val_dataset = ImageFolderDataset(root=val_root, transform=val_transform)

        # Wrap the datasets with PyTorchDataset to support the .iterator() method.
        train_loader = PyTorchDataset(
            dataset=train_dataset,
            device=config.device,
            prepare_batch=lambda batch, device: Batch(*batch).to(device),
            num_classes=8,
        )
        val_loader = PyTorchDataset(
            dataset=val_dataset,
            device=config.device,
            prepare_batch=lambda batch, device: Batch(*batch).to(device),
            num_classes=8,
        )

        return train_loader, val_loader


    else:
        raise RuntimeError(f"Unknown dataset: {config.data_name}")
    
    return BaseLoader(train_dataset), BaseLoader(val_dataset)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, dataset, preprocess, aug, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd
        self.aug = aug

    def __getitem__(self, i):
        print(f"Loading sample {i} augmix")
        x, y = self.dataset[i]
        if self.no_jsd:
            return self.aug(x, self.preprocess), y
        else:
            im_tuple = (
                self.preprocess(x),
                self.aug(x, self.preprocess),
                self.aug(x, self.preprocess),
            )
            return im_tuple, y

    def __len__(self):
        return len(self.dataset)
