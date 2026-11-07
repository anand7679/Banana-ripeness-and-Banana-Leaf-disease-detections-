#file-->>dataset & transforms
# Grid creation: The model takes an input image and cuts it into a grid of non-overlapping, 
# fixed-size square tiles (for example, 16 × 16 pixels each).Flattening: Each 2D patch is flattened 
# into a 1D vector. For an RGB color patch of size 16 × 16,
# this means \(16 \times 16 \times 3 \text{ (color channels)} = 768\) raw values




import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image

IMG_SIZE = 224

# convert original folder names to binary class index
ORIG_TO_BINARY = {
    "ripe": 1,        # BestTE -> label 1
    "unripe": 0,      # NotBestTE -> label 0
    "overripe": 0,
    "rotten": 0
}

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.15,0.15,0.15,0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])

class BinaryImageFolder(Dataset):
    """
    Wrap torchvision.datasets.ImageFolder but remap original class names
    to binary labels using ORIG_TO_BINARY.
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # iterate subfolders
        for class_name in sorted(os.listdir(root)):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            mapped = ORIG_TO_BINARY.get(class_name.lower())
            if mapped is None:
                # skip unknown folders
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg','.jpeg','.png')):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(mapped)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def create_dataloaders(data_root, batch_size=32, num_workers=4):
    train_ds = BinaryImageFolder(os.path.join(data_root, "train"), transform=get_transforms(True))
    val_ds = BinaryImageFolder(os.path.join(data_root, "valid"), transform=get_transforms(False))
    test_ds = BinaryImageFolder(os.path.join(data_root, "test"), transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ["NotBestToEat", "BestToEat"]
    return train_loader, val_loader, test_loader, classes
