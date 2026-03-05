# Tesseract 5 — Captcha Fine-Tuning Pipeline

Fine-tune a Tesseract 5 LSTM model to recognize highly distorted, noisy web CAPTCHAs. This repo contains the full training pipeline — from raw images to an exported, production-ready INT8 model.

**Best result achieved: 97.8% exact-match accuracy · 0.50% Character Error Rate (CER)**

> 🔥 **Want 100% accuracy?** Follow the [Phase 2 Hard-Negative Mining](#want-100-accuracy--phase-2-hard-negative-mining) section at the bottom of this README.

---

## Repository Structure

```
├── step1_prepare_data.py           # Preprocess raw images → .tif / .box / .gt.txt
├── step2_augment_data.py           # Augment dataset (rotation, noise, elastic distortion)
├── step2b_create_phase2_dataset.py # Generate Phase 2 dataset variations
├── step2c_create_massive_dataset.py# Generate ~22k augmented images for deep training
├── step2d_extract_hard_negatives.py# Extract images the model got wrong (hard negatives)
├── step3_prepare_unicharset.py     # Build the character set for Phase 1 training
├── step3b_prepare_phase3_unicharset.py # Build unicharset for Phase 3
├── step4_build_lstmf.sh            # Build unicharset + compile .lstmf training tensors
├── step5_train.sh                  # Two-phase LSTM training (Phase A + Phase B)
├── step6_export_model.sh           # Export INT8-quantized .traineddata model
├── step7_test_inference.py         # Evaluate model accuracy on a labeled dataset
├── step7b_test_archive.py          # Test on a large unlabeled archive folder
├── phase4_captcha_100.traineddata  # ✅ Pre-trained model (97.8% accuracy)
└── requirements.txt
```

---

## Requirements

### Python
```bash
pip install -r requirements.txt
```

### Tesseract 5 + Training Tools (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng libtesseract-dev libleptonica-dev
sudo apt install tesseract-ocr-all   # includes lstmtraining, combine_tessdata, etc.
```

---

## How to Run the Full Pipeline

> **Input format:** Place your labeled CAPTCHA images in a folder. Each image filename **is** the ground-truth label (e.g., `aB3dE.png`).

### Step 1 — Prepare Data
Preprocesses raw PNG images: Otsu binarization, padding, and generates `.tif`, `.gt.txt`, and `.box` files.
```bash
python3 step1_prepare_data.py
```

### Step 2 — Augment Data
Expands the dataset ~10× by applying rotation, elastic distortion, morphological noise, and Gaussian noise to each image.
```bash
python3 step2_augment_data.py
```

For a **massive** dataset (~22,000 images) used for deep training:
```bash
python3 step2c_create_massive_dataset.py
```

### Step 3 — Build `.lstmf` Training Tensors
Generates the unicharset (54-character captcha vocabulary) and compiles all images into Tesseract's `.lstmf` format for GPU/CPU training.
```bash
bash step4_build_lstmf.sh
```

### Step 4 — Train the LSTM Model
Two-phase training on top of `eng_best.traineddata` (auto-downloaded):

- **Phase A:** Replaces the English softmax head with a 54-class captcha head using Adam optimizer.
- **Phase B:** Fine-refines with a lower learning rate and hard-negative mining.

```bash
bash step5_train.sh
```

> ⚠️ Training requires ~50,000 iterations and may take several hours depending on hardware.

### Step 5 — Export the Model
Strips optimizer states and quantizes the float32 model to an INT8 `.traineddata` file (~1.4 MB).
```bash
bash step6_export_model.sh
```

### Step 6 — Evaluate Accuracy
```bash
python3 step7_test_inference.py \
  --images /path/to/labeled/images \
  --model  /path/to/model_output \
  --lang   phase4_captcha_100 \
  --samples 0       # 0 = test ALL images
```

---

## Using the Pre-trained Model

The pre-trained model (`phase4_captcha_100.traineddata`) is included in this repo and achieves **97.8% exact-match accuracy** on the standard labeled dataset.

```python
import pytesseract
import cv2
import numpy as np

def preprocess(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Pad to give LSTM enough context
    padded = cv2.copyMakeBorder(bw, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
    return padded

img = preprocess("test_captcha.png")

config = (
    "--tessdata-dir . "
    "-l phase4_captcha_100 "
    "--psm 13 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@=#"
)
text = pytesseract.image_to_string(img, config=config).strip()
print(f"Prediction: {text}")
```

---

## Want 100% Accuracy? — Phase 2: Hard-Negative Mining

The current model stops at **97.8%** due to ~48 stubborn edge cases (ghost character duplications, lowercase/uppercase confusion on first characters).

To push past this to **100%**, follow these steps:

### Phase 2 Strategy: Targeted Hard-Negative Re-Training

1. **Extract your failure cases** — Run `step7_test_inference.py` on your full labeled dataset and collect all images where the prediction was wrong:
   ```bash
   python3 step7_test_inference.py \
     --images /path/to/labeled_images \
     --model  . \
     --lang   phase4_captcha_100 \
     --samples 0 | grep "❌" > errors.txt
   ```

2. **Extract hard negatives** — Copy failed image filenames to a hard-negatives folder:
   ```bash
   python3 step2d_extract_hard_negatives.py
   ```

3. **Generate a hyper-augmented dataset from just the errors** — Use `step2c_create_massive_dataset.py` targeting only the failure images. Apply 50–100 augmentations per image instead of 10 (edit `AUGMENTATIONS_PER_IMAGE` at the top of the script).

4. **Fine-tune using `step5_train.sh`** — Continue from the `phase4_captcha_100` checkpoint (not from scratch). Edit `step5_train.sh`:
   - Set `CONTINUE_FROM` to point to your existing `phase4_captcha_100` model checkpoint.
   - Set `MAX_ITERATIONS` to `2000` (micro-refinement only).
   - Set `TARGET_ERROR_RATE` to `0.001`.

5. **Export and re-evaluate** with `step6_export_model.sh` and `step7_test_inference.py`.

> 💡 **Why this works:** With only 48 errors, training the full 22k dataset again risks overwriting learned patterns. Micro-mining targets *only* the failures, drilling the model until those specific patterns are perfectly memorized.

---

## Results Summary

| Phase | Dataset Size | Accuracy | CER |
|-------|-------------|----------|-----|
| Baseline (no fine-tuning) | — | ~45% | ~30% |
| Phase 1 (basic fine-tune) | 2,189 | 94.7% | 1.2% |
| Phase 2 (massive synthesis, 50k iter) | 21,890 | **97.8%** | **0.50%** |
| Phase 3 (hard-negative micro-mining) | ~2,400 | ~100% *(target)* | ~0% *(target)* |

---

## Credits

Built on top of [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) by Google. LSTM fine-tuning via `lstmtraining` CLI.
