import os
import glob
import cv2
import random
import multiprocessing
from pathlib import Path
import numpy as np
from PIL import Image

# Configuration
INPUT_DIR = "/home/ca/Downloads/lable_captcha"
OUTPUT_DIR = "/home/ca/Projects/captcha_model/training_data_phase4"
TARGET_AUGMENTATIONS_PER_IMAGE = 10
PAD_Y, PAD_X = 25, 25

# Tesseract resolution for boxing
DPI = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)

def elastic_transform(image, alpha=20, sigma=3, alpha_affine=1):
    """Elastic deformation of images as described in [Simard2003]."""
    random_state = np.random.RandomState(None)
    shape = image.shape
    shape_size = shape[:2]

    if len(shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        shape = image.shape

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    dx = cv2.GaussianBlur((random_state.rand(shape[0], shape[1]) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(shape[0], shape[1]) * 2 - 1), (0, 0), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)

    distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return distorted_image

def apply_augmentation(img_bin, aug_type):
    """Applies specific mathematical morphological distortions for variation."""
    img_rgb = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2RGB)
    
    if aug_type == 'erode':
        kernel = np.ones((2,2), np.uint8)
        # Erase white space -> fattens black text
        img_res = cv2.erode(img_bin, kernel, iterations=1)
    elif aug_type == 'dilate':
        kernel = np.ones((2,2), np.uint8)
        # Dilate white space -> thins black text
        img_res = cv2.dilate(img_bin, kernel, iterations=1)
    elif aug_type == 'elastic':
        img_res = elastic_transform(img_rgb, alpha=15, sigma=4)
        img_res = cv2.cvtColor(img_res, cv2.COLOR_RGB2GRAY)
    elif aug_type == 'blur':
        img_res = cv2.GaussianBlur(img_bin, (3,3), 0)
    elif aug_type == 'noise':
        noise = np.random.randint(0, 2, img_bin.shape, dtype=np.uint8) * 255
        # Salt and pepper on white background, focusing on making some black pixels white or white black randomly
        mask = np.random.rand(*img_bin.shape) < 0.05
        img_bin_copy = img_bin.copy()
        img_bin_copy[mask] = noise[mask]
        img_res = img_bin_copy
    elif aug_type == 'rotation':
        angle = random.uniform(-3, 3)
        h, w = img_bin.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        img_res = cv2.warpAffine(img_bin, M, (w, h), borderValue=255)
    else:
        # Original clean
        img_res = img_bin
        
    return img_res

def generate_box_file(img_path, ground_truth, h, w):
    """Generates Tesseract compatible .box pseudo-coordinates for a single line image."""
    box_path = img_path.replace(".tif", ".box")
    chars = list(ground_truth)
    num_chars = len(chars)
    if num_chars == 0:
        return
        
    char_width = w // num_chars
    
    with open(box_path, 'w', encoding='utf-8') as f:
        for i, char in enumerate(chars):
            left = i * char_width
            right = (i + 1) * char_width if i < num_chars - 1 else w
            bottom = 0
            top = h
            # Format: <char> <left> <bottom> <right> <top> <page>
            f.write(f"{char} {left} {bottom} {right} {top} 0\n")

def process_single_image(img_path):
    """Worker function to process and augment one original image into multiple variants."""
    try:
        filename = os.path.basename(img_path)
        ground_truth = os.path.splitext(filename)[0]
        base_name = ground_truth.replace(" ", "_").replace("/", "").replace("\\", "")

        # Read & Preprocess
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return 0
        _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Apply strict training padding
        img_padded = cv2.copyMakeBorder(
            img_bin, PAD_Y, PAD_Y, PAD_X, PAD_X,
            cv2.BORDER_CONSTANT, value=255
        )
        
        augmentations = ['original', 'erode', 'dilate', 'elastic', 'elastic', 'blur', 'noise', 'rotation', 'rotation', 'original']
        
        generated_count = 0
        for i, aug_type in enumerate(augmentations[:TARGET_AUGMENTATIONS_PER_IMAGE]):
            # Apply distortion
            img_aug = apply_augmentation(img_padded, aug_type)
            
            out_prefix = os.path.join(OUTPUT_DIR, f"{base_name}_aug{i}_{aug_type}")
            tif_file = f"{out_prefix}.tif"
            gt_file = f"{out_prefix}.gt.txt"
            
            # Save TIF with DPI
            pil_img = Image.fromarray(img_aug)
            pil_img.save(tif_file, dpi=(DPI, DPI))
            
            # Save GT Text
            with open(gt_file, 'w', encoding='utf-8') as f:
                f.write(ground_truth)
                
            # Generate Box
            h, w = img_aug.shape
            generate_box_file(tif_file, ground_truth, h, w)
            
            generated_count += 1
            
        return generated_count
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return 0

def main():
    print(f"Scanning for images in '{INPUT_DIR}'...")
    png_images = glob.glob(os.path.join(INPUT_DIR, "*.png"))
    jpg_images = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    all_images = png_images + jpg_images
    
    total_images = len(all_images)
    print(f"Found {total_images} original images. Target augmentations per image: {TARGET_AUGMENTATIONS_PER_IMAGE}")
    print(f"Expected generated images: {total_images * TARGET_AUGMENTATIONS_PER_IMAGE}")
    
    print(f"Using {multiprocessing.cpu_count()} CPU cores for parallel synthesis...")
    
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.imap_unordered(process_single_image, all_images)
    
    total_generated = 0
    for i, count in enumerate(results):
        total_generated += count
        if (i+1) % 50 == 0 or (i+1) == total_images:
            print(f"Processed original images: {i+1}/{total_images} | Generated dataset: {total_generated}")
            
    pool.close()
    pool.join()
    
    print(f"\nSuccessfully built Phase 4 Massive Dataset: {total_generated} images in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
