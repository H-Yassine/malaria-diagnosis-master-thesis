import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset import MalariaDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--labels-csv", type=str, default="labels.csv")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # in case of gpu (pc-pool), mac gpu (mps) as backup, else just cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("using device:", device)

    # imagenet stats since we use pretrained resnet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # augmentation only for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize
    ])

    # no augmentation for val, just resize + normalize
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = MalariaDataset(args.labels_csv, args.data_root, "train", transform=train_transform)
    val_dataset = MalariaDataset(args.labels_csv, args.data_root, "val", transform=val_transform)

    # num_workers 0 (for my laptop)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # pretrained resnet50, swap last layer for our 2 classes
    model = models.resnet50(weights="IMAGENET1K_V2")
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_dataset)

        # check val acc every epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"epoch {epoch+1}/{args.epochs} - train_loss: {train_loss:.4f} - val_acc: {val_acc:.4f}")

        # save only if its the best so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"saved new best model, val_acc: {val_acc:.4f}")
        # save the latest epoch !! In case the training stops :)
        last_checkpoint_path = os.path.join(args.checkpoint_dir, "last_epoch.pth")
        torch.save(model.state_dict(), last_checkpoint_path)
    print("training done. best val acc:", best_val_acc)

if __name__ == "__main__":
    main()
