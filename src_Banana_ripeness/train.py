# training 
import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from dataset import create_dataloaders
from model import build_vit
from sklearn.metrics import accuracy_score

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    preds = []
    targets = []
    for imgs, labels in tqdm(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
        targets.extend(labels.cpu().numpy().tolist())
    acc = accuracy_score(targets, preds)
    return sum(losses)/len(losses), acc

def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())
    acc = accuracy_score(targets, preds)
    return sum(losses)/len(losses), acc

def main():
    data_root = "Banana Ripeness Classification Dataset"
    epochs = 10
    batch_size = 16
    lr = 3e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader, classes = create_dataloaders(data_root, batch_size=batch_size)
    model = build_vit(model_name="vit_small_patch16_224", num_classes=2, pretrained=True)
    model = model.to(device)

    class_counts = torch.bincount(
        torch.as_tensor(train_loader.dataset.labels), minlength=len(classes)
    ).float()
    class_weights = class_counts.sum() / (len(classes) * class_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch} | Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"checkpoints/best_vit.pth")
            print("Saved best model")

    # final test
    model.load_state_dict(torch.load("checkpoints/best_vit.pth"))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test loss {test_loss:.4f} acc {test_acc:.4f}")

if __name__ == "__main__":
    main()
