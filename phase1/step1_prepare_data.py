#!/usr/bin/env python3
"""
Phase 1: Data Preparation Pipeline
===================================
Reads raw 203x50 RGB captcha PNGs from input_dir.
For each image:
  - Converts to grayscale and applies Otsu's binarization
  - Pads 25px white border on all sides (→ 253x100)
  - Saves .tif, .gt.txt, and WordStr .box files to output_dir

Then splits ALL .lstmf paths (after Phase 2) 85/15 into train.list / eval.list.
The list files are written to model_output/ directory.
"""

import os
import cv2
import glob
import random
import argparse
from pathlib import Path
import sys

IMAGE_DIR = "/home/ca/Projects/captcha_model/lable_captcha (1)/lable_captcha"
TRAIN_DIR  = "/home/ca/Projects/captcha_model/training_data"
OUTPUT_DIR = "/home/ca/Projects/captcha_model/model_output"

PAD_Y = 25
PAD_X = 25


def sanitize_label(label: str) -> str:
    """
    Filenames may contain special filesystem characters that are valid captcha
    chars (@, =, .).  Return as-is; these are all valid UTF-8 and Tesseract
    will handle them once in the unicharset.
    """
    return label


def preprocess_captcha_dataset(input_dir: str, output_dir: str) -> list:
    """
    Main data-preparation loop.
    Returns list of (base_name, ground_truth) tuples.
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(glob.glob(str(input_path / "*.png")))
    if not image_files:
        print(f"[ERROR] No PNG files found in {input_dir}")
        sys.exit(1)

    processed = []
    skipped   = 0

    for img_path in image_files:
        filename     = os.path.basename(img_path)
        # FIX: os.path.splitext returns a tuple; take [0] for the stem
        ground_truth = sanitize_label(os.path.splitext(filename)[0])

        # ── Load and binarize ────────────────────────────────────────────────
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"  [WARN] Could not read {img_path} — skipping")
            skipped += 1
            continue

        _, img_bin = cv2.threshold(
            img_gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        # ── Pad with white border ─────────────────────────────────────────────
        img_padded = cv2.copyMakeBorder(
            img_bin, PAD_Y, PAD_Y, PAD_X, PAD_X,
            cv2.BORDER_CONSTANT, value=255
        )
        height, width = img_padded.shape

        # ── Output paths ──────────────────────────────────────────────────────
        # Use a sanitized safe filename for disk (replace @ = . with safe chars
        # for the filesystem key, but keep the real label in gt.txt / box)
        safe_stem   = ground_truth.replace("/", "_")          # just in case
        base_output = output_path / safe_stem

        # .tif (required by tesseract lstm.train)
        tif_path = str(base_output) + ".tif"
        cv2.imwrite(tif_path, img_padded)

        # .gt.txt (plain ground truth — used by unicharset_extractor)
        with open(str(base_output) + ".gt.txt", "w", encoding="utf-8") as f:
            f.write(ground_truth)

        # .box (WordStr format — Tesseract bottom-left origin)
        # Format: WordStr <left> <bottom> <right> <top> <page> #<text> \t <left> <bottom> <right> <top> <page>
        box_line = (
            f"WordStr 0 0 {width} {height} 0 #{ground_truth}\n"
            f"\t 0 0 {width} {height} 0\n"
        )
        with open(str(base_output) + ".box", "w", encoding="utf-8") as f:
            f.write(box_line)

        processed.append(safe_stem)

    print(f"\n{'='*60}")
    print(f"  Processed : {len(processed)} images")
    print(f"  Skipped   : {skipped} images")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*60}\n")
    return processed


def write_list_files(stems: list, output_dir: str, train_dir: str) -> None:
    """
    Write model_output/train.list and model_output/eval.list
    pointing to .lstmf files that Phase 2 will generate.
    (85 / 15 split)
    """
    random.seed(42)
    shuffled = stems[:]
    random.shuffle(shuffled)

    split_idx  = int(len(shuffled) * 0.85)
    train_list = shuffled[:split_idx]
    eval_list  = shuffled[split_idx:]

    model_out = Path(output_dir)
    model_out.mkdir(parents=True, exist_ok=True)

    train_file = model_out / "train.list"
    eval_file  = model_out / "eval.list"

    with open(train_file, "w") as f:
        for stem in train_list:
            f.write(f"{train_dir}/{stem}.lstmf\n")

    with open(eval_file, "w") as f:
        for stem in eval_list:
            f.write(f"{train_dir}/{stem}.lstmf\n")

    print(f"  train.list : {len(train_list)} entries → {train_file}")
    print(f"  eval.list  : {len(eval_list)}  entries → {eval_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1: Prepare captcha dataset for Tesseract LSTM training"
    )
    parser.add_argument("--input",  default=IMAGE_DIR,  help="Directory with raw PNG captchas")
    parser.add_argument("--output", default=TRAIN_DIR,  help="Output directory for .tif/.gt.txt/.box")
    parser.add_argument("--model-output", default=OUTPUT_DIR, help="Directory for train/eval lists")
    args = parser.parse_args()

    print(f"\n[Phase 1] Data Preparation")
    print(f"  Input : {args.input}")
    print(f"  Output: {args.output}\n")

    stems = preprocess_captcha_dataset(args.input, args.output)
    write_list_files(stems, args.model_output, args.output)

    print("\n[Phase 1] COMPLETE — run 02_build_unicharset_and_lstmf.sh next.\n")
