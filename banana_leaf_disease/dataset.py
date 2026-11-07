# Dataset and transform 
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = 224
CLASS_NAMES = ("cordana", "healthy", "sigatoka")

_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
_NORMALIZE_STD = (0.229, 0.224, 0.225)


def build_transforms(training: bool) -> transforms.Compose:
    """Build the ImageNet-compatible transform pipeline."""
    operations = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15,
                    hue=0.05,
                ),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(_NORMALIZE_MEAN, _NORMALIZE_STD),
        ]
    )
    return transforms.Compose(operations)


def _load_split(root: Path, split: str, training: bool) -> datasets.ImageFolder:
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split does not exist: {split_dir}")

    dataset = datasets.ImageFolder(
        str(split_dir), transform=build_transforms(training)
    )
    discovered_classes = tuple(name.lower() for name in dataset.classes)
    if discovered_classes != CLASS_NAMES:
        raise ValueError(
            f"Expected classes {CLASS_NAMES} in {split_dir}, "
            f"but found {discovered_classes}."
        )
    return dataset


def create_dataloaders(
    data_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test loaders from a dataset directory."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    root = Path(data_root)
    train_dataset = _load_split(root, "train", training=True)
    val_dataset = _load_split(root, "val", training=False)
    test_dataset = _load_split(root, "test", training=False)

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_args),
        DataLoader(val_dataset, shuffle=False, **loader_args),
        DataLoader(test_dataset, shuffle=False, **loader_args),
    )
