"""
This script creates one CSV file for all images from the three datasets that I got from NLM:
uninfected, vivax and falciparum. For every image, the class, binary label
and patient ID are saved.

The dataset is split into three parts: training, validation and test. the split is based on the
patients, not on images. This is important because images from
the same patient should not appear in different parts. Otherwise, data
leakage could occur and the test results might be unrealistically high.

The output file is labels.csv with the following columns:
    image_path, class_name, label, patient_id, split

Labels:
    0 = uninfected
    1 = infected
"""
import random
import csv
from pathlib import Path

# ---------------------------------------------------------------------
# paths - change DATA_ROOT !!! this path is on my private laptop
#
# ---------------------------------------------------------------------
# AI helped me add the data-root argument so I can use the same script on different computers without changing the path.
DATA_ROOT = Path("/Users/yassine.hermak/Desktop/Masterarbeit/malaria-diagnosis/data/Datasets/NLM")
OUTPUT_CSV = Path("labels.csv")
####

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# ---------------------------------------------------------------------
# each class has its own folder structure :)
# ---------------------------------------------------------------------
CLASS_CONFIG = {
    "uninfected": {
        "images_glob": "NIH-NLM-ThickBloodSmears-Uninfected/Uninfected Patients/*/tiled/*.tiff",
        "label": 0,
        "patient_from_parts_index": -3,  # .../Uninfected Patients/PATIENT/tiled/img.tiff
    },
    "vivax": {
        "images_glob": "NIH-NLM-ThickBloodSmearsPV-Vivax/All_PvTk/*/*.jpg",
        "label": 1,
        "patient_from_parts_index": -2,  # .../All_PvTk/PATIENT/img.jpg
    },
    "falciparum": {
        "images_glob": "ThickBloodSmears_150-Falciparum/*/*.jpg",
        "label": 1,
        "patient_from_parts_index": -2,  # .../PATIENT/img.jpg
    },
}


def collect_images_for_class(class_name, config):
    """Collects all images of one class and reads the patient ID from the folder name."""
   
    pattern = config["images_glob"]
    matches = list(DATA_ROOT.glob(pattern))

    if not matches:
        print(f"  [!] no images found for '{class_name}', check the glob pattern: {pattern}")

    rows = []
    for img_path in matches:
        parts = img_path.parts
        patient_id = parts[config["patient_from_parts_index"]]
        rows.append({
            "image_path": str(img_path),
            "class_name": class_name,
            "label": config["label"],
            "patient_id": patient_id,
        })
    return rows


def split_patients(patient_ids, seed):
    """
    Splits the patient IDs into training, validation and test sets.The split is done separately for each class.
    """
    patient_ids = list(patient_ids)
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    ## remaining patients go to the test set
    n_test = n - n_train - n_val

    train_ids = set(patient_ids[:n_train])
    val_ids = set(patient_ids[n_train:n_train + n_val])
    test_ids = set(patient_ids[n_train + n_val:])

    return train_ids, val_ids, test_ids, (n_train, n_val, n_test)


def main():
    print("building labels file...")
    print(f"data root: {DATA_ROOT}")

    all_rows = []
    for class_name, config in CLASS_CONFIG.items():
        print(f"\n- collecting images for class: {class_name}")
        rows = collect_images_for_class(class_name, config)
        print(f"  found {len(rows)} images")
        all_rows.extend(rows)

    print(f"\ntotal images collected: {len(all_rows)}")

    # collect patient IDs for each class
    patients_per_class = {}
    for class_name in CLASS_CONFIG:
        ids = {r["patient_id"] for r in all_rows if r["class_name"] == class_name}
        patients_per_class[class_name] = ids

    split_assignment = {}  # patient_id -> "train"/"val"/"test"

    print("\nsplitting patients (not images!) into train/val/test:")
    for class_name, ids in patients_per_class.items():
        train_ids, val_ids, test_ids, counts = split_patients(ids, RANDOM_SEED)
        print(f"  {class_name:12s} -> patients: train={counts[0]}, val={counts[1]}, test={counts[2]}")

        for pid in train_ids:
            split_assignment[pid] = "train"
        for pid in val_ids:
            split_assignment[pid] = "val"
        for pid in test_ids:
            split_assignment[pid] = "test"

    # add the split to each image
    for row in all_rows:
        row["split"] = split_assignment[row["patient_id"]]

    # count images per split
    split_image_counts = {"train": 0, "val": 0, "test": 0}
    for row in all_rows:
        split_image_counts[row["split"]] += 1
    print(f"\nimages per split (final): {split_image_counts}")

    # write everything to csv
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "class_name", "label", "patient_id", "split"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\ndone. labels file saved to: {OUTPUT_CSV.resolve()}")



if __name__ == "__main__":
    main()
