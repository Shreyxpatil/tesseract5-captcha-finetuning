#!/usr/bin/env python3
"""
Phase 5: Inference Test & Evaluation
======================================
Applies the SAME binarize+pad preprocessing used in training,
then runs pytesseract with --psm 13 on the trained model.

Computes per-sample accuracy and overall Character Error Rate (CER).

Usage:
    python3 05_test_inference.py [--samples N] [--model MODEL_DIR]
"""

import os
import sys
import glob
import random
import argparse
import subprocess
from pathlib import Path

try:
    import cv2
except ImportError:
    print("Installing opencv-python-headless...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "opencv-python-headless", "--break-system-packages", "-q"])
    import cv2

try:
    import pytesseract
except ImportError:
    print("Installing pytesseract...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "pytesseract", "--break-system-packages", "-q"])
    import pytesseract

import numpy as np

# ── Defaults ─────────────────────────────────────────────────────────────────
IMAGE_DIR   = "/home/ca/Projects/captcha_model/lable_captcha (1)/lable_captcha"
MODEL_DIR   = "/home/ca/Projects/captcha_model/model_output"
MODEL_NAME  = "captcha_final"
PAD_Y, PAD_X = 25, 25


def preprocess_for_inference(img_path: str) -> np.ndarray:
    """Apply the same binarize+pad pipeline used during training."""
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise RuntimeError(f"Cannot read: {img_path}")
    _, img_bin = cv2.threshold(img_gray, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_padded = cv2.copyMakeBorder(
        img_bin, PAD_Y, PAD_Y, PAD_X, PAD_X,
        cv2.BORDER_CONSTANT, value=255
    )
    return img_padded


def char_error_rate(pred: str, truth: str) -> float:
    """Levenshtein-based CER = edit_distance / len(truth)."""
    n, m = len(truth), len(pred)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    dp = list(range(n + 1))
    for j in range(1, m + 1):
        prev = dp.copy()
        dp[0] = j
        for i in range(1, n + 1):
            cost = 0 if truth[i-1] == pred[j-1] else 1
            dp[i] = min(dp[i-1] + 1, prev[i] + 1, prev[i-1] + cost)
    return dp[n] / n


def run_evaluation(image_dir: str, model_dir: str, model_name: str,
                   n_samples: int, use_eval_list: bool) -> None:
    # ── Find images to test ───────────────────────────────────────────────────
    if use_eval_list:
        eval_list_path = Path(model_dir) / "eval.list"
        if eval_list_path.exists():
            # map lstmf paths back to PNG filenames
            lstmf_lines = [l.strip() for l in eval_list_path.read_text().splitlines() if l.strip()]
            stems = [Path(l).stem for l in lstmf_lines]
            image_files = []
            for stem in stems:
                matches = glob.glob(f"{image_dir}/{stem}.png")
                if not matches:
                    # handle safe-stem mapping (@ = . might be in name)
                    matches = glob.glob(f"{image_dir}/{stem}.png")
                if matches:
                    image_files.append(matches[0])
            print(f"  Using eval.list: {len(image_files)} images found")
        else:
            print("  eval.list not found — sampling randomly from full dataset")
            use_eval_list = False

    if not use_eval_list:
        all_images = sorted(glob.glob(f"{image_dir}/*.png"))
        random.seed(42)
        image_files = random.sample(all_images, min(n_samples, len(all_images)))
        print(f"  Randomly sampled: {len(image_files)} images")

    if not image_files:
        print("ERROR: No images found to test.")
        sys.exit(1)

    # ── Check if model exists ─────────────────────────────────────────────────
    model_file = Path(model_dir) / f"{model_name}.traineddata"
    if not model_file.exists():
        # Try parent folder (Tesseract wants tessdata dir → model_name.traineddata)
        print(f"\n[WARN] {model_file} not found.")
        print("  Falling back to testing with default 'eng' model for comparison.")
        tessdata_dir = None
        lang = "eng"
    else:
        tessdata_dir = model_dir
        lang = model_name
        print(f"\n  Model: {model_file}")

    # ── Tesseract config ──────────────────────────────────────────────────────
    tess_config = "--psm 13"
    if tessdata_dir:
        tess_config += f" --tessdata-dir {tessdata_dir}"

    print(f"  Config: {tess_config}")
    print(f"  Lang  : {lang}")
    print()

    # ── Run inference ─────────────────────────────────────────────────────────
    results = []
    HEADER = f"{'#':>4}  {'Ground Truth':<12}  {'Predicted':<12}  {'Match':^7}  {'CER':>6}"
    print(HEADER)
    print("-" * len(HEADER))

    for i, img_path in enumerate(image_files[:n_samples]):
        filename    = os.path.basename(img_path)
        ground_truth = os.path.splitext(filename)[0]

        try:
            img_proc = preprocess_for_inference(img_path)

            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, img_proc)

            predicted = pytesseract.image_to_string(
                tmp_path,
                lang=lang,
                config=tess_config
            ).strip().replace("\n", "").replace(" ", "")
            os.unlink(tmp_path)

        except Exception as e:
            predicted = f"[ERR: {e}]"

        cer    = char_error_rate(predicted, ground_truth)
        match  = "✅" if predicted == ground_truth else "❌"
        results.append((ground_truth, predicted, cer))
        print(f"{i+1:>4}  {ground_truth:<12}  {predicted:<12}  {match:^7}  {cer:>6.3f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results)
    if n == 0:
        print("\nNo results to summarize.")
        return

    exact_matches = sum(1 for gt, pred, _ in results if gt == pred)
    mean_cer      = sum(c for _, _, c in results) / n
    exact_acc     = exact_matches / n * 100

    print()
    print("=" * 55)
    print(f"  Samples tested    : {n}")
    print(f"  Exact matches     : {exact_matches}/{n}  ({exact_acc:.1f}%)")
    print(f"  Mean CER          : {mean_cer:.4f}  ({mean_cer*100:.2f}%)")
    print("=" * 55)

    if exact_acc >= 90:
        print("  🎉 Excellent accuracy (≥90%)!")
    elif exact_acc >= 75:
        print("  ✅ Good accuracy — consider more iterations for improvement.")
    else:
        print("  ⚠️  Accuracy below 75% — review training logs or increase iterations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tesseract Captcha Inference Test")
    parser.add_argument("--images",   default=IMAGE_DIR,  help="Path to raw PNG directory")
    parser.add_argument("--model",    default=MODEL_DIR,  help="Tessdata directory (containing .traineddata)")
    parser.add_argument("--lang",     default=MODEL_NAME, help="Tesseract language/model name")
    parser.add_argument("--samples",  type=int, default=200, help="Number of images to test")
    parser.add_argument("--eval-list",action="store_true",  help="Use eval.list for held-out testing")
    args = parser.parse_args()

    print("\n[Phase 5] Inference Evaluation")
    print(f"  Image dir : {args.images}")
    print(f"  Model dir : {args.model}")
    print(f"  Model name: {args.lang}")
    run_evaluation(args.images, args.model, args.lang,
                   args.samples, args.eval_list)
