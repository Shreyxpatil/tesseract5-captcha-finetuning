#!/usr/bin/env python3
"""
Phase 1b: Data Augmentation
===================================
Applies 4 distinct transformations to every .tif image in training_data/.
Augmented copies will be saved in the same Directory.

The 4 Augmentations:
1. Rotation: -3 to +3 degrees (uniform).
2. Elastic/Wave distortion: slight ripples along X and Y axes.
3. Erosion/Dilation (Morphological): thickens or thins characters.
4. Additive Gaussian Noise: injects static while preserving binarization.
"""

import os
import cv2
import glob
import math
import random
import argparse
import numpy as np
from pathlib import Path

TRAIN_DIR = "/home/ca/Projects/captcha_model/training_data"
OUTPUT_DIR = "/home/ca/Projects/captcha_model/model_output"

def rot_image(img, angle):
    """Rotate image by `angle` degrees (positive = counter-clockwise)."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 255 border value (white) since OCR expects white background
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

def wave_distortion(img, magnitude=1.5, frequency=0.08):
    """Apply a slight sinusoidal wave distortion."""
    h, w = img.shape[:2]
    distorted = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            offset_x = int(magnitude * math.sin(2 * math.pi * i * frequency))
            offset_y = int(magnitude * math.cos(2 * math.pi * j * frequency))
            if i + offset_y < h and j + offset_x < w and i + offset_y >= 0 and j + offset_x >= 0:
                distorted[i, j] = img[i + offset_y, j + offset_x]
            else:
                distorted[i, j] = 255 # white background
    return distorted

def morph_distortion(img):
    """Randomly apply either slight erosion or dilation."""
    kernel = np.ones((2, 2), np.uint8)
    if random.choice([True, False]):
        # Erosion (black chars get thicker in our binarized inverted space, wait - OpenCV erosion on white bg thins the black chars)
        # 1-iteration is standard
        return cv2.erode(img, kernel, iterations=1)
    else:
        # Dilation (black chars get thinner)
        return cv2.dilate(img, kernel, iterations=1)

def add_noise(img, ratio=0.02):
    """Add salt & pepper or gaussian-like noise to the binarized image."""
    h, w = img.shape[:2]
    noisy = np.copy(img)
    num_noise = int(h * w * ratio)
    for _ in range(num_noise):
        y, x = random.randint(0, h - 1), random.randint(0, w - 1)
        noisy[y, x] = random.choice([0, 255])
    return noisy

def augment_dataset(train_dir: str):
    """Apply the 4 augmentations to every original .tif image in the directory."""
    input_path = Path(train_dir)
    
    # Only grab the originals (those that don't have _aug in their name)
    tif_files = [f for f in glob.glob(str(input_path / "*.tif")) if "_aug_" not in f]
    
    print(f"Found {len(tif_files)} original images. Augmenting x4...")
    
    processed = 0
    stems = []
    
    for count, tif_path in enumerate(tif_files, 1):
        bg_path = Path(tif_path)
        stem = bg_path.stem
        stems.append(stem) # append original
        
        img = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # Also need ground truth to duplicate .gt.txt and .box
        gt_path = input_path / f"{stem}.gt.txt"
        box_path = input_path / f"{stem}.box"
        
        if not gt_path.exists() or not box_path.exists():
            continue
            
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_text = f.read()
            
        with open(box_path, 'r', encoding='utf-8') as f:
            box_text = f.read()
            
        # Aug 1: Rotation
        angle = random.uniform(-3, 3)
        img_rot = rot_image(img, angle)
        
        # Aug 2: Wave
        img_wave = wave_distortion(img, magnitude=random.uniform(1.0, 2.0))
        
        # Aug 3: Morphological
        img_morph = morph_distortion(img)
        
        # Aug 4: Noise
        img_noise = add_noise(img, ratio=random.uniform(0.01, 0.03))
        
        augs = [
            ("aug_rot", img_rot),
            ("aug_wav", img_wave),
            ("aug_mrp", img_morph),
            ("aug_nse", img_noise)
        ]
        
        for suffix, aug_img in augs:
            new_stem = f"{stem}_{suffix}"
            stems.append(new_stem)
            
            # Save TIF
            cv2.imwrite(str(input_path / f"{new_stem}.tif"), aug_img)
            
            # Save duplicates of GT and BOX
            with open(input_path / f"{new_stem}.gt.txt", 'w', encoding='utf-8') as f:
                f.write(gt_text)
                
            with open(input_path / f"{new_stem}.box", 'w', encoding='utf-8') as f:
                f.write(box_text)
                
        processed += 1
        if processed % 200 == 0:
            print(f"  Augmented {processed}/{len(tif_files)}...")

    return stems

def write_list_files(stems: list, output_dir: str, train_dir: str) -> None:
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

    print(f"  train.list : {len(train_list)} entries")
    print(f"  eval.list  : {len(eval_list)} entries")


if __name__ == "__main__":
    print("\n[Phase 1b] Data Augmentation")
    stems = augment_dataset(TRAIN_DIR)
    
    print("\nRe-generating train.list and eval.list for 10,000+ files...")
    write_list_files(stems, OUTPUT_DIR, TRAIN_DIR)
    
    print("\n[Phase 1b] COMPLETE — run 02_build_unicharset_and_lstmf.sh next.\n")
