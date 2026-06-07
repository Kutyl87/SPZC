from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CIFAR100WithIds(datasets.CIFAR100):
    def __init__(self, split, **kwargs):
        super().__init__(train=split == "train", **kwargs)
        self.split = split

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        sample_id = f"{self.split}:{index}"
        return image, label, sample_id


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


def get_cifar100_loaders(data_dir, batch_size=64, num_workers=0):
    transform = build_transforms()

    train_dataset = CIFAR100WithIds(
        split="train",
        root=data_dir,
        download=True,
        transform=transform,
    )
    test_dataset = CIFAR100WithIds(
        split="test",
        root=data_dir,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader
