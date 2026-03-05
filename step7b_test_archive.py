import os
import glob
import cv2
import pytesseract
import tempfile
import multiprocessing
from functools import partial
import sys
import time

IMAGE_DIR = "/home/ca/Downloads/archive"
MODEL_DIR = "/home/ca/Projects/captcha_model/model_output"
MODEL_NAME = "captcha_final"
PAD_Y, PAD_X = 25, 25

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
        return ground_truth, "[ERR_READ]", False
        
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
    return ground_truth, predicted, match

def main():
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    
    if not images:
        print(f"No JPG images found in {IMAGE_DIR}")
        return

    total = len(images)
    print(f"Found {total} images in {IMAGE_DIR}\n")
    print(f"Using {multiprocessing.cpu_count()} CPU cores for inference...\n")
    
    exact_matches = 0
    
    # Process in parallel
    start_time = time.time()
    
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    # Use imap_unordered for a responsive progress bar
    results = pool.imap_unordered(process_image, images)
    
    # Only print first few to avoid console spam for 100k+ images
    print_limit = 20
    
    for i, (ground_truth, predicted, match) in enumerate(results):
        if match:
            exact_matches += 1
            
        # Update progress bar
        sys.stdout.write(f"\rProcessed: {i+1}/{total} ({(i+1)/total*100:.1f}%) | Matches: {exact_matches}")
        sys.stdout.flush()
        
    pool.close()
    pool.join()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    accuracy = (exact_matches / total) * 100
    
    print("\n\n" + "=" * 40)
    print(f"Total Images : {total}")
    print(f"Exact Matches: {exact_matches}")
    print(f"Accuracy     : {accuracy:.2f}%")
    print(f"Time Taken   : {elapsed:.2f} seconds ({elapsed/60:.2f} min)")
    print(f"Rate         : {total/elapsed:.2f} images/sec")
    print("=" * 40)

if __name__ == "__main__":
    main()
