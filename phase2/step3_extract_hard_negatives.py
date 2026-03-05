import os
import shutil
import glob
import cv2
import pytesseract
import tempfile
import multiprocessing
from functools import partial
import sys
import time

IMAGE_DIR = "/home/ca/Downloads/lable_captcha"
MODEL_DIR = "/home/ca/Projects/captcha_model/model_output"
MODEL_NAME = "captcha_final"
PAD_Y, PAD_X = 25, 25

# Output directory for hard negatives
HARD_NEG_DIR = "/home/ca/Projects/captcha_model/hard_negatives"
os.makedirs(HARD_NEG_DIR, exist_ok=True)

# Configure Tesseract
TESS_CONFIG = f"--psm 13 --tessdata-dir {MODEL_DIR}"

def preprocess_for_inference(img_path: str):
    """Apply binarization and padding as done during training inference."""
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None
    _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_padded = cv2.copyMakeBorder(
        img_bin, PAD_Y, PAD_Y, PAD_X, PAD_X,
        cv2.BORDER_CONSTANT, value=255
    )
    return img_padded

def process_image(img_path):
    """Process a single image and return its prediction."""
    filename = os.path.basename(img_path)
    ground_truth = os.path.splitext(filename)[0]
    
    img_proc = preprocess_for_inference(img_path)
    if img_proc is None:
        return img_path, ground_truth, "[ERR_READ]", False
        
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, img_proc)
    
    try:
        predicted = pytesseract.image_to_string(
            tmp_path,
            lang=MODEL_NAME,
            config=TESS_CONFIG
        ).strip().replace("\n", "").replace(" ", "")
    except Exception as e:
        predicted = f"[ERR]"
        
    os.unlink(tmp_path)
    
    match = (predicted == ground_truth)
    return img_path, ground_truth, predicted, match

def main():
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    if not images:
        images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
    
    if not images:
        print(f"No images found in {IMAGE_DIR}")
        return

    total = len(images)
    print(f"Generating predictions for {total} images to extract hard negatives...")
    
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.imap_unordered(process_image, images)
    
    hard_negatives_count = 0
    
    with open(os.path.join(HARD_NEG_DIR, "hard_negatives_log.txt"), "w") as log_file:
        for i, (img_path, ground_truth, predicted, match) in enumerate(results):
            sys.stdout.write(f"\rProcessed: {i+1}/{total} | Found errors: {hard_negatives_count}")
            sys.stdout.flush()
            
            if not match:
                hard_negatives_count += 1
                
                # Copy image to hard negatives directory
                filename = os.path.basename(img_path)
                dest_path = os.path.join(HARD_NEG_DIR, filename)
                shutil.copy2(img_path, dest_path)
                
                # Create ground truth file
                gt_path = os.path.join(HARD_NEG_DIR, f"{ground_truth}.gt.txt")
                with open(gt_path, "w") as f:
                    f.write(ground_truth)
                    
                # Log the error
                log_file.write(f"{filename}: predicted '{predicted}', truth '{ground_truth}'\n")
        
    pool.close()
    pool.join()
    
    print(f"\n\nIsolated {hard_negatives_count} hard negative images.")
    print(f"Files saved to: {HARD_NEG_DIR}")

if __name__ == "__main__":
    main()
