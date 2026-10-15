"""
Dataset preparation script.
==============================
Sorts labeled video datasets into the data/real/ and data/fake/ folders
that dataset.py and train.py expect.

Supports MULTIPLE DATASETS merged together -- just run this script once
per dataset source you download, pointing at different --source-dir /
--metadata each time. Since everything lands in the same data/real and
data/fake folders, train.py automatically trains on the union of every
dataset you've added, no code changes needed. This is how you actually
get closer to "as accurate as possible": more, more varied, real-world
labeled examples beats any single trick or heuristic tweak.

Currently supports two metadata formats out of the box:

  1. DFDC-style `metadata.json`:
       {
         "aagfhgtpmv.mp4": {"label": "FAKE", "split": "train", ...},
         "aayfryxljh.mp4": {"label": "REAL", "split": "train", ...},
         ...
       }
     (This is exactly the format used by the Kaggle "Deepfake Detection
     Challenge" sample_videos.zip and full dataset.)

  2. A simple CSV with two columns: filename,label (label = REAL or FAKE).
     Useful for datasets that don't ship a metadata.json, or for your own
     hand-labeled clips.

Usage:
    # DFDC-style metadata.json
    python prepare_dataset.py --source-dir ./downloads/dfdc_sample --metadata metadata.json

    # CSV-based labels
    python prepare_dataset.py --source-dir ./downloads/my_dataset --csv labels.csv

Videos are COPIED (not moved) by default so your original download stays
intact -- pass --move if you want to save disk space instead.
"""

import os
import json
import csv
import shutil
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Sort a labeled video dataset into data/real and data/fake")
    p.add_argument("--source-dir", type=str, required=True,
                    help="Folder containing the video files for this dataset")
    p.add_argument("--metadata", type=str, default=None,
                    help="Path to a DFDC-style metadata.json (relative to source-dir if not absolute)")
    p.add_argument("--csv", type=str, default=None,
                    help="Path to a CSV file with columns: filename,label")
    p.add_argument("--data-root", type=str, default="./data",
                    help="Destination root containing real/ and fake/ subfolders")
    p.add_argument("--move", action="store_true",
                    help="Move files instead of copying (saves disk space, original dataset folder is emptied)")
    p.add_argument("--prefix", type=str, default="",
                    help="Optional filename prefix to avoid collisions when merging multiple "
                         "datasets that might reuse the same filenames (e.g. --prefix dfdc_)")
    return p.parse_args()


def load_labels_from_metadata_json(source_dir, metadata_path):
    path = metadata_path if os.path.isabs(metadata_path) else os.path.join(source_dir, metadata_path)
    with open(path) as f:
        meta = json.load(f)

    labels = {}
    for filename, info in meta.items():
        label = str(info.get("label", "")).upper()
        if label in ("REAL", "FAKE"):
            labels[filename] = label
    return labels


def load_labels_from_csv(csv_path):
    labels = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header row if present
        if header and header[0].lower() not in ("filename", "file"):
            # first row wasn't actually a header -- treat it as data
            if len(header) >= 2:
                labels[header[0]] = header[1].strip().upper()
        for row in reader:
            if len(row) < 2:
                continue
            filename, label = row[0].strip(), row[1].strip().upper()
            if label in ("REAL", "FAKE"):
                labels[filename] = label
    return labels


def main():
    args = parse_args()

    if not args.metadata and not args.csv:
        raise SystemExit("Provide either --metadata (DFDC-style JSON) or --csv (filename,label CSV).")

    if args.metadata:
        labels = load_labels_from_metadata_json(args.source_dir, args.metadata)
    else:
        labels = load_labels_from_csv(args.csv)

    if not labels:
        raise SystemExit("No REAL/FAKE labels found -- check the metadata/csv format matches what's expected.")

    real_dir = os.path.join(args.data_root, "real")
    fake_dir = os.path.join(args.data_root, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    copied, missing = 0, 0
    real_count, fake_count = 0, 0

    for filename, label in labels.items():
        src_path = os.path.join(args.source_dir, filename)
        if not os.path.exists(src_path):
            missing += 1
            continue

        dest_dir = real_dir if label == "REAL" else fake_dir
        dest_name = f"{args.prefix}{filename}" if args.prefix else filename
        dest_path = os.path.join(dest_dir, dest_name)

        if args.move:
            shutil.move(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)

        copied += 1
        if label == "REAL":
            real_count += 1
        else:
            fake_count += 1

    print(f"\nDone. {copied} videos sorted ({real_count} real, {fake_count} fake).")
    if missing:
        print(f"WARNING: {missing} entries in the labels file had no matching video "
              f"in {args.source_dir} and were skipped.")

    total_real = len(os.listdir(real_dir))
    total_fake = len(os.listdir(fake_dir))
    print(f"\ndata/real/ now has {total_real} videos total")
    print(f"data/fake/ now has {total_fake} videos total")

    ratio = total_fake / max(total_real, 1)
    if ratio > 3 or ratio < 0.33:
        print(f"\nNOTE: your real:fake ratio is quite imbalanced ({total_real}:{total_fake}). "
              f"train.py now applies class-weighted loss automatically to compensate, but "
              f"adding more of the minority class (if you can find another dataset source) "
              f"would still help more than the re-weighting alone.")


if __name__ == "__main__":
    main()
