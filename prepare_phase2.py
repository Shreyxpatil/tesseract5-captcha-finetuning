#!/usr/bin/env python3
import os
import cv2
import glob
import random
from pathlib import Path

INPUT_DIR = "/home/ca/Projects/captcha_model/phase2_dataset"
TRAIN_DIR = "/home/ca/Projects/captcha_model/training_data_phase2"
OUTPUT_DIR = "/home/ca/Projects/captcha_model/model_output_phase2"

PAD_Y = 25
PAD_X = 25

def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(TRAIN_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Grabbing both .png and .jpg
    image_files = sorted(glob.glob(str(input_path / "*.png")) + glob.glob(str(input_path / "*.jpg")) + glob.glob(str(input_path / "*.jpeg")))
    
    if not image_files:
        print(f"[ERROR] No images found in {INPUT_DIR}")
        return

    processed = []
    skipped = 0

    for img_path in image_files:
        filename = os.path.basename(img_path)
        stem = os.path.splitext(filename)[0]
        
        # Read the ground truth from the already generated .gt.txt
        gt_file = input_path / f"{stem}.gt.txt"
        if not gt_file.exists():
            print(f"  [WARN] Missing GT for {img_path} — skipping")
            skipped += 1
            continue
            
        with open(gt_file, "r") as f:
            ground_truth = f.read().strip()
            
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            skipped += 1
            continue

        _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_padded = cv2.copyMakeBorder(img_bin, PAD_Y, PAD_Y, PAD_X, PAD_X, cv2.BORDER_CONSTANT, value=255)
        height, width = img_padded.shape

        safe_stem = stem.replace("/", "_")
        base_output = output_path / safe_stem

        # .tif
        cv2.imwrite(str(base_output) + ".tif", img_padded)

        # .gt.txt
        with open(str(base_output) + ".gt.txt", "w", encoding="utf-8") as f:
            f.write(ground_truth)

        # .box
        box_line = f"WordStr 0 0 {width} {height} 0 #{ground_truth}\n\t 0 0 {width} {height} 0\n"
        with open(str(base_output) + ".box", "w", encoding="utf-8") as f:
            f.write(box_line)

        processed.append(safe_stem)

    print(f"Processed: {len(processed)} images. Skipped: {skipped}")

    # Write train/eval lists
    random.seed(42)
    shuffled = processed[:]
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * 0.85)
    train_list = shuffled[:split_idx]
    eval_list = shuffled[split_idx:]

    with open(Path(OUTPUT_DIR) / "train.list", "w") as f:
        for s in train_list:
            f.write(f"{TRAIN_DIR}/{s}.lstmf\n")

    with open(Path(OUTPUT_DIR) / "eval.list", "w") as f:
        for s in eval_list:
            f.write(f"{TRAIN_DIR}/{s}.lstmf\n")

    print(f"train.list: {len(train_list)} | eval.list: {len(eval_list)}")

if __name__ == "__main__":
    main()
