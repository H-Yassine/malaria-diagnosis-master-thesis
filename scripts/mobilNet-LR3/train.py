import argparse
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt

from dataset import MalariaDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--labels-csv", type=str, default="labels.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.00003)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("using device:", device)

    # imagenet stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # resize + augmentation
    train_transform = transforms.Compose([
        transforms.Resize((56, 56)),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize((56, 56)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = MalariaDataset(args.labels_csv, args.data_root, "train", transform=train_transform)
    val_dataset = MalariaDataset(args.labels_csv, args.data_root, "val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # pretrained mobilenet_v3_small, classifier head is replaced for our 2 classes
    model = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model = model.to(device)

    class_weights = torch.tensor([2.0, 1.0]).to(device)  # [uninfected, infected] - weight uninfected more :)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    #  history of train loss and val accuracy for creating csv and plotting
    train_losses = []
    val_accs = []

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

        train_losses.append(train_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"saved new best model, val_acc: {val_acc:.4f}")

    print("training done. best val acc:", best_val_acc)

    # creating the csv file for the training history
    with open("training_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_acc"])
        for i in range(len(train_losses)):
            writer.writerow([i + 1, train_losses[i], val_accs[i]])

    # plotting the training loss
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.title("training loss over epochs")
    plt.savefig("train_loss_plot.png")
    plt.close()

    # plotting the validation accuracy
    plt.figure()
    plt.plot(range(1, len(val_accs) + 1), val_accs)
    plt.xlabel("epoch")
    plt.ylabel("val accuracy")
    plt.title("validation accuracy over epochs")
    plt.savefig("val_acc_plot.png")
    plt.close()

    print("saved: training_history.csv, train_loss_plot.png, val_acc_plot.png")

if __name__ == "__main__":
    main()
