# I built this Grad-CAM script based on the pytorch-grad-cam library
# https://github.com/jacobgil/pytorch-grad-cam


import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from dataset import MalariaDataset

DATA_ROOT = "/Users/yassine.hermak/Desktop/Masterarbeit/malaria-diagnosis/data/Datasets"
CHECKPOINT = "checkpoints/best_model.pth"
LABELS_CSV = "labels.csv"
OUTPUT_DIR = "gradcam_outputs"

N_PER_GROUP = {"FN": 5, "FP": 5, "TP": 5, "TN": 5}

# same resolution we used in LR3
RESOLUTION = 56

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu")
    print("using device:", device)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # reducing the resolution
    degrade_transform = transforms.Compose([
        transforms.Resize((RESOLUTION, RESOLUTION)),
        transforms.Resize((224, 224))
    ])

    # degradation + tensor + normalize, same as our model input
    full_transform = transforms.Compose([
        degrade_transform,
        transforms.ToTensor(),
        normalize
    ])

    test_dataset = MalariaDataset(LABELS_CSV, DATA_ROOT, "test", transform=full_transform)
    loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # the different image-groups we need: FN, FP, TP, Tn // first N of each group
    groups = {"FN": [], "FP": [], "TP": [], "TN": []}
    for i in range(len(all_labels)):
        label = all_labels[i]
        pred = all_preds[i]

        if label == 1 and pred == 0:
            group = "FN"
        elif label == 0 and pred == 1:
            group = "FP"
        elif label == 1 and pred == 1:
            group = "TP"
        else:
            group = "TN"

        if len(groups[group]) < N_PER_GROUP[group]:
            groups[group].append(i)

    for g, idxs in groups.items():
        print(f"{g}: {len(idxs)} images selected")

    # I used the last convolutional block as recommended for ResNet from the original Grad-CAM paper by Selvaraju et al.
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    for group_name, indices in groups.items():
        for idx in indices:
            row = test_dataset.df.iloc[idx]
            img_path = row["image_path"]
            raw_image = Image.open(img_path).convert("RGB")

            # use the same degraded image for model and also as background for the heatmap
            degraded_image = degrade_transform(raw_image)

            input_tensor = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])(degraded_image).unsqueeze(0).to(device)

            vis_image = np.array(degraded_image).astype(np.float32) / 255.0

            predicted_class = all_preds[idx]
            targets = [ClassifierOutputTarget(predicted_class)]

            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
            overlay = show_cam_on_image(vis_image, grayscale_cam, use_rgb=True)

            out_name = f"{group_name}_{idx}.png"
            Image.fromarray(overlay).save(os.path.join(OUTPUT_DIR, out_name))
            print(f"saved {out_name}")

    print(f"\ndone. outputs in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
