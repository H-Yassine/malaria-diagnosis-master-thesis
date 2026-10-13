import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
 
from dataset import MalariaDataset


#   The use of argparse was first seen in Kaggle scripts and was then implemented with support from Claude based on the parameters already used in this project. The --data-root argument was also introduced to avoid a fixed dataset path and make the script easier to reuse on different systems.
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--labels-csv", type=str, default="labels_uganda.csv")
    parser.add_argument("--checkpoint", type=str, default="/Users/yassine.hermak/Desktop/Masterarbeit/malaria-diagnosis/scripts/Resnet50-LR3/checkpoints/best_model.pth")
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
        transforms.Resize((56, 56)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
 
    test_dataset = MalariaDataset(args.labels_csv, args.data_root, "test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
 
    # same model architecture -> load trained weights
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
 
    all_preds = []
    all_labels = []
    all_probs = []  # probability of infected class(roc-auc) -- just for reference
 
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
 
            probs = torch.softmax(outputs, dim=1)[:, 1]  # prob of class 1 (infected)
            _, preds = torch.max(outputs, 1)
 
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
 
    # this dataset is infected-only (all labels = 1), so I force labels=[0, 1] to keep the confusion matrix 2x2 even when class 0 never appears. otherwise .ravel() below would fail if every prediction also happened to be class 1.
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    print("\nconfusion matrix: rows=actual[uninfected,infected], columns=predicted")
    print(cm)
 
    tn, fp, fn, tp = cm.ravel()
 
    #  tn and fp will always be 0 since there is no actual uninfected class present, so accuracy == sensitivity
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # recall for infected class
    support_infected = tp + fn
 
    print(f"accuracy: {accuracy:.4f}")
    print(f"sensitivity (recall for infected): {sensitivity:.4f}")
    print(f"support - infected: {support_infected}")
    print("\nnote: this external set contains infected images only -- specificity, precision,")
    print("f1-score, and roc-auc are not defined without an uninfected class present, so they")
    print("are intentionally omitted here rather than computed as misleading trivial values.")
 
    report = classification_report(
        all_labels, all_preds, labels=[0, 1], target_names=["uninfected", "infected"], zero_division=0
    )
    print("\nfull report:")
    print(report)
 
    with open("test_results_uganda.txt", "a") as f:
        f.write(f"\n--- external test (Uganda), checkpoint: {args.checkpoint} ---\n")
        f.write(f"confusion matrix:\n{cm}\n")
        f.write(f"accuracy: {accuracy:.4f}\n")
        f.write(f"sensitivity: {sensitivity:.4f}\n")
        f.write(f"support - infected: {support_infected}\n")
        f.write("specificity/precision/f1/roc-auc: not applicable (infected-only external set)\n")
        f.write(report)
        f.write("\n")
 
    # confusion matrix plot
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("confusion matrix (external test - Uganda)")
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.xticks([0, 1], ["uninfected", "infected"])
    plt.yticks([0, 1], ["uninfected", "infected"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar()
    plt.savefig("confusion_matrix_uganda.png")
    plt.close()
 
    # roc curve intentionally omitted: roc_auc_score/roc_curve require both classes which is not possible in this test
 
    print("\nsaved: test_results_uganda.txt, confusion_matrix_uganda.png")
 
if __name__ == "__main__":
    main()
 
