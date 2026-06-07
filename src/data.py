from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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


def build_datasets(dataset_name, data_dir):
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
