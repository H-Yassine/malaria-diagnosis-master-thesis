import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from dataset import MalariaDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--labels-csv", type=str, default="labels.csv")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()

def main():
    args = get_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("using device:", device)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = MalariaDataset(args.labels_csv, args.data_root, "test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # same model architecture -> load trained weights
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []  # probability of infected class(roc-auc)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:, 1]  # prob of class 1 (infected)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # confusion matrix: rows = correct info, columnss = predicted
    cm = confusion_matrix(all_labels, all_preds)
    print("\nconfusion matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)  # recall for infected class
    specificity = tn / (tn + fp)  # recall for uninfected class
    precision = tp / (tp + fp)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    support_uninfected = tn + fp
    support_infected = tp + fn

    auc = roc_auc_score(all_labels, all_probs)

    print(f"accuracy: {accuracy:.4f}")
    print(f"sensitivity (recall for infected): {sensitivity:.4f}")
    print(f"specificity (recall for uninfected): {specificity:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"f1-score: {f1:.4f}")
    print(f"support - uninfected: {support_uninfected}, infected: {support_infected}")
    print(f"roc-auc: {auc:.4f}")

    report = classification_report(all_labels, all_preds, target_names=["uninfected", "infected"])
    print("\nfull report:")
    print(report)

    with open("test_results.txt", "a") as f:
        f.write(f"\n--- run using checkpoint: {args.checkpoint} ---\n")
        f.write(f"confusion matrix:\n{cm}\n")
        f.write(f"accuracy: {accuracy:.4f}\n")
        f.write(f"sensitivity: {sensitivity:.4f}\n")
        f.write(f"specificity: {specificity:.4f}\n")
        f.write(f"precision: {precision:.4f}\n")
        f.write(f"f1-score: {f1:.4f}\n")
        f.write(f"support - uninfected: {support_uninfected}, infected: {support_infected}\n")
        f.write(f"roc-auc: {auc:.4f}\n")
        f.write(report)
        f.write("\n")

    # confusion matrix plot
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("confusion matrix")
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.xticks([0, 1], ["uninfected", "infected"])
    plt.yticks([0, 1], ["uninfected", "infected"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # roc curve plot
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"auc = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("roc curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.close()

    print("\nsaved: test_results.txt, confusion_matrix.png, roc_curve.png")

if __name__ == "__main__":
    main()
