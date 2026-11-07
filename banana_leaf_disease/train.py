# Train the ViT banana leaf disease classifier
import argparse
import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch import optim
from tqdm import tqdm
from .dataset import CLASS_NAMES, create_dataloaders
from .model import CHECKPOINT_PATH, MODEL_NAME, build_vit


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    predictions = []
    targets = []
    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
        targets.extend(labels.cpu().numpy().tolist())
    return sum(losses) / len(losses), accuracy_score(targets, predictions)


def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    predictions = []
    targets = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())
    return sum(losses) / len(losses), accuracy_score(targets, predictions)


def train(
    data_root="Banana_Leaf_Dataset",
    checkpoint_path=CHECKPOINT_PATH,
    epochs=1,
    batch_size=16,
    learning_rate=3e-4,
    pretrained=True,
    device=None,
):
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root, batch_size=batch_size
    )
    model = build_vit(
        model_name=MODEL_NAME,
        num_classes=len(CLASS_NAMES),
        pretrained=pretrained,
    ).to(selected_device)

    class_counts = torch.bincount(
        torch.as_tensor(train_loader.dataset.targets),
        minlength=len(CLASS_NAMES),
    ).float()
    if torch.any(class_counts == 0):
        raise ValueError("Every disease class must have at least one training image.")
    class_weights = class_counts.sum() / (len(CLASS_NAMES) * class_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(selected_device))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, selected_device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, selected_device)
        print(
            f"Epoch {epoch} | Train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print("Saved best model")

    model.load_state_dict(torch.load(checkpoint_path, map_location=selected_device))
    test_loss, test_acc = validate(model, test_loader, criterion, selected_device)
    print(f"Test loss {test_loss:.4f} acc {test_acc:.4f}")


if __name__ == "__main__":
    data_root = "Banana_Leaf_Dataset"
    checkpoint_path = CHECKPOINT_PATH
    epochs = 10
    batch_size = 16
    learning_rate = 3e-4
    pretrained = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(
        data_root=data_root,
        checkpoint_path=checkpoint_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pretrained=pretrained,
        device=device,
    )
