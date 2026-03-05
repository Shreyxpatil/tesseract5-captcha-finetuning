import os
import shutil
import glob
import random
import cv2
import numpy as np

HARD_NEG_DIR = "/home/ca/Projects/captcha_model/hard_negatives"
LABLE_CAPTCHA_DIR = "/home/ca/Downloads/lable_captcha"
PHASE2_DIR = "/home/ca/Projects/captcha_model/phase2_dataset"

os.makedirs(PHASE2_DIR, exist_ok=True)

def apply_augmentation(img):
    """Applies a random lightweight augmentation to an image."""
    h, w = img.shape[:2]
    
    # Randomly choose augmentation type
    aug_type = random.choice(['rotate', 'noise', 'blur', 'morph'])
    
    if aug_type == 'rotate':
        # Slight rotation (-3 to 3 degrees)
        angle = random.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
        
    elif aug_type == 'noise':
        # Light Gaussian noise
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = img_noisy
        
    elif aug_type == 'blur':
        # Mild blur
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
    elif aug_type == 'morph':
        # Erosion or Dilation (very mild)
        kernel = np.ones((2, 2), np.uint8)
        if random.random() > 0.5:
            img = cv2.erode(img, kernel, iterations=1)
        else:
            img = cv2.dilate(img, kernel, iterations=1)
            
    return img

def main():
    print(f"Creating Phase 2 dataset in {PHASE2_DIR}...")
    
    hard_negatives = glob.glob(os.path.join(HARD_NEG_DIR, "*.*"))
    # Filter out text files
    hard_negatives = [f for f in hard_negatives if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(hard_negatives)} hard negatives.")
    
    # 1. Augment Hard Negatives (10x each)
    augmented_count = 0
    for img_path in hard_negatives:
        filename = os.path.basename(img_path)
        stem = os.path.splitext(filename)[0]
        ground_truth = stem # Based on how lable_captcha is named
        
        img = cv2.imread(img_path)
        if img is None: continue
            
        # Copy original
        dest_orig = os.path.join(PHASE2_DIR, f"hn_orig_{filename}")
        cv2.imwrite(dest_orig, img)
        with open(os.path.join(PHASE2_DIR, f"hn_orig_{stem}.gt.txt"), "w") as f:
            f.write(ground_truth)
        augmented_count += 1
            
        # Generate 9 augmentations
        for i in range(9):
            aug_img = apply_augmentation(img.copy())
            dest_aug = os.path.join(PHASE2_DIR, f"hn_aug{i}_{filename}")
            cv2.imwrite(dest_aug, aug_img)
            with open(os.path.join(PHASE2_DIR, f"hn_aug{i}_{stem}.gt.txt"), "w") as f:
                f.write(ground_truth)
            augmented_count += 1
            
    print(f"Generated {augmented_count} augmented hard negatives.")
    
    # 2. Add random correct images (500)
    all_source_images = glob.glob(os.path.join(LABLE_CAPTCHA_DIR, "*.*"))
    all_source_images = [f for f in all_source_images if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Find which source images were hard negatives
    hard_neg_filenames = set(os.path.basename(p) for p in hard_negatives)
    
    # Filter out hard negatives
    correct_images = [f for f in all_source_images if os.path.basename(f) not in hard_neg_filenames]
    
    print(f"Found {len(correct_images)} correct images.")
    
    random.seed(42)
    sample_size = min(500, len(correct_images))
    sampled_correct = random.sample(correct_images, sample_size)
    
    for img_path in sampled_correct:
        filename = os.path.basename(img_path)
        stem = os.path.splitext(filename)[0]
        ground_truth = stem
        
        # Copy image
        dest_path = os.path.join(PHASE2_DIR, f"correct_{filename}")
        shutil.copy2(img_path, dest_path)
        
        # Write ground truth
        with open(os.path.join(PHASE2_DIR, f"correct_{stem}.gt.txt"), "w") as f:
            f.write(ground_truth)
            
    print(f"Added {sample_size} random correct images.")
    print(f"Total Phase 2 dataset size: {augmented_count + sample_size}")

if __name__ == "__main__":
    main()
