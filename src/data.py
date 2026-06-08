from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch

class CIFAR100WithIds(datasets.CIFAR100):
    def __init__(self, split, **kwargs):
        super().__init__(train=split == "train", **kwargs)
        self.split = split

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, f"{self.split}:{index}"


class ImagenetteWithIds(datasets.Imagenette):
    def __init__(self, split, **kwargs):
        super().__init__(split=split, **kwargs)
        self.split = split

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, f"{self.split}:{index}"


def build_transforms():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

# provide split ratios if you want to split imagenette dataset it should look like [2 : 1]
# this means it will split data set into 2 : 1 ratios
def build_datasets(dataset_name, data_dir, split_ratios=None):
    transform = build_transforms()

    if dataset_name == "cifar100":
        train_dataset = CIFAR100WithIds(
            split="train",
            root=data_dir,
            download=True,
            transform=transform,
        )
        eval_dataset = CIFAR100WithIds(
            split="test",
            root=data_dir,
            download=True,
            transform=transform,
        )
        return train_dataset, eval_dataset

    if dataset_name == "imagenette":
        train_dataset = ImagenetteWithIds(
            split="train",
            root=data_dir,
            size="160px",
            download=True,
            transform=transform,
        )
        eval_dataset = ImagenetteWithIds(
            split="val",
            root=data_dir,
            size="160px",
            download=True,
            transform=transform,
        )

        if split_ratios is not None and isinstance(split_ratios, list) and len(split_ratios)==2:
            total_size = len(train_dataset)
            # split dataset into given ratios
            train_size = int(total_size * (split_ratios[0] / sum(split_ratios)))           
            unlabelled_size = total_size - train_size  
            
            generator = torch.Generator().manual_seed(42)
            
            partial_train_dataset, unlabelled_dataset = random_split(
                train_dataset, 
                [train_size, unlabelled_size],
                generator=generator
            )
            
            return partial_train_dataset, unlabelled_dataset, eval_dataset

        return train_dataset, eval_dataset

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_data_loaders(dataset_name, data_dir, batch_size=64, num_workers=0):
    train_dataset, eval_dataset = build_datasets(dataset_name, data_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, eval_loader


def get_data_loaders_with_split(dataset_name, data_dir, batch_size=64, num_workers=0, split_ratios=[4, 1]):
    train_dataset, unlabelled_dataset, eval_dataset = build_datasets(dataset_name, data_dir, split_ratios)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    unlabelled_loader = DataLoader(
        unlabelled_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, unlabelled_loader, eval_loader

