import argparse
import os
import csv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="labels_uganda.csv")
    return parser.parse_args()

def main():
    args = get_args()

    image_files = sorted(f for f in os.listdir(args.image_dir)
                          if f.lower().endswith(".jpg"))

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label", "split"])
        for fname in image_files:
            writer.writerow([fname, 1, "test"])

    print(f"wrote {len(image_files)} entries to {args.output_csv}")

if __name__ == "__main__":
    main()
