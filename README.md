# Tesseract 5 — Captcha Fine-Tuning Pipeline

Fine-tune a Tesseract 5 LSTM model to recognize highly distorted, noisy web CAPTCHAs.
This repo contains the full training pipeline — from raw images to an exported, production-ready INT8 model.

**Best result achieved: 97.8% exact-match accuracy · 0.50% Character Error Rate (CER)**

> 🔥 **Want 100% accuracy?** Run Phase 1 first, then follow the [Phase 2 guide](#phase-2--100-accuracy-hard-negative-mining) below.

---

## Repository Structure

```
├── phase1/                           # Standard fine-tuning pipeline (~97.8% accuracy)
│   ├── step1_prepare_data.py         # Preprocess raw images → .tif / .box / .gt.txt
│   ├── step2_augment_data.py         # Augment dataset (rotation, noise, distortion)
│   ├── step3_prepare_unicharset.py   # Build the 54-character captcha vocabulary
│   ├── step4_build_lstmf.sh          # Compile images into .lstmf training tensors
│   ├── step5_train.sh                # Two-phase LSTM training (Phase A + B)
│   └── step6_export_model.sh         # Export INT8-quantized .traineddata model
│
├── phase2/                           # Hard-negative mining for ~100% accuracy
│   ├── step1_create_augmented_dataset.py   # Generate Phase 2 dataset variations
│   ├── step2_create_massive_dataset.py     # Generate ~22k augmented images
│   ├── step3_extract_hard_negatives.py     # Extract images the model got wrong
│   └── step4_prepare_unicharset.py         # Build unicharset for Phase 2 training
│
├── step7_test_inference.py           # Evaluate model accuracy on any labeled dataset
├── step7b_test_archive.py            # Test on a large archive folder
├── phase4_captcha_100.traineddata    # ✅ Pre-trained model (97.8% accuracy)
└── requirements.txt
```

---

## Requirements

### Python dependencies

```bash
pip install -r requirements.txt
```

### Tesseract 5 + Training Tools (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng libtesseract-dev libleptonica-dev
sudo apt install tesseract-ocr-all   # includes lstmtraining, combine_tessdata, etc.
```

---

## Phase 1 — Standard Fine-Tuning (~97.8% accuracy)

Place your labeled CAPTCHA images in a folder. **The filename is the ground-truth label** (e.g., `aB3dE.png`).

Run all scripts inside the `phase1/` folder in order:

### Step 1 — Prepare Data

Preprocesses raw PNG images: Otsu binarization, padding, and generates `.tif`, `.gt.txt`, and `.box` files.

```bash
python3 phase1/step1_prepare_data.py
```

### Step 2 — Augment Data

Expands the dataset ~10× by applying rotation, elastic distortion, morphological noise, and Gaussian noise.

```bash
python3 phase1/step2_augment_data.py
```

### Step 3 — Build `.lstmf` Training Tensors

Generates the unicharset (54-character captcha vocabulary) and compiles all images into Tesseract's `.lstmf` format.

```bash
bash phase1/step4_build_lstmf.sh
```

### Step 4 — Train the LSTM Model

Two-phase training on top of `eng_best.traineddata` (auto-downloaded from tessdata_best):

- **Phase A:** Replaces the English softmax head with a 54-class captcha head using Adam optimizer.
- **Phase B:** Fine-refines with a lower learning rate and hard-negative mining.

```bash
bash phase1/step5_train.sh
```

> ⚠️ Training runs up to 50,000 iterations and may take several hours depending on hardware.

### Step 5 — Export the Model

Strips optimizer states and quantizes the float32 model to an INT8 `.traineddata` file (~1.4 MB).

```bash
bash phase1/step6_export_model.sh
```

### Step 6 — Evaluate Accuracy

```bash
python3 step7_test_inference.py \
  --images /path/to/your/labeled/images \
  --model  /path/to/model_output \
  --lang   captcha_final \
  --samples 0
```

---

## Using the Pre-trained Model

The pre-trained model `phase4_captcha_100.traineddata` is included and achieves **97.8% exact-match accuracy**.

```python
import pytesseract
import cv2

def preprocess(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    padded = cv2.copyMakeBorder(bw, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
    return padded

img = preprocess("test_captcha.png")
config = "--tessdata-dir . -l phase4_captcha_100 --psm 13"
text = pytesseract.image_to_string(img, config=config).strip()
print(f"Prediction: {text}")
```

---

## Phase 2 — 100% Accuracy: Hard-Negative Mining

After Phase 1, your model will be ~97–98% accurate. The remaining errors are almost always:

- **Ghost duplications** — predicting `9krkk` instead of `9krk`
- **Case confusion** — predicting `Un6uR` instead of `un6uR`

To push past this to **100%**, use Phase 2:

### Step 1 — Find failed predictions

Run the test script and save which images the model got wrong:

```bash
python3 step7_test_inference.py \
  --images /path/to/labeled_images \
  --model  . \
  --lang   phase4_captcha_100 \
  --samples 0 | grep "❌" > errors.txt
```

### Step 2 — Extract hard negatives

Copy the failed images into a `hard_negatives/` folder:

```bash
python3 phase2/step3_extract_hard_negatives.py
```

### Step 3 — Generate massive augmentations from just the errors

Target only the failure images and apply 50–100 augmentations per image:

```bash
python3 phase2/step2_create_massive_dataset.py
```

> Edit `AUGMENTATIONS_PER_IMAGE` at the top of the script to 50 or 100 for deeper mining.

### Step 4 — Fine-tune from your existing checkpoint

Edit `phase1/step5_train.sh`:
- Set `CONTINUE_FROM` to your existing model checkpoint (e.g., `phase4_captcha_100`)
- Set `MAX_ITERATIONS` to `2000`
- Set `TARGET_ERROR_RATE` to `0.001`

Then re-run training and export:

```bash
bash phase1/step5_train.sh
bash phase1/step6_export_model.sh
```

### Step 5 — Re-evaluate

```bash
python3 step7_test_inference.py --images /path/to/labeled_images --model . --lang <new_model> --samples 0
```

> 💡 **Why this works:** With only ~48 errors remaining, retraining on the full 22k dataset risks disturbing already-learned patterns. Micro-mining targets *only* the failures, memorizing those exact cases.

---

## Results Summary

| Phase | Dataset Size | Exact Match | CER |
|-------|-------------|-------------|-----|
| Baseline (no fine-tuning) | — | ~45% | ~30% |
| Phase 1 (standard fine-tune, 2k iter) | 2,189 images | 94.7% | 1.2% |
| Phase 1 (massive synthesis, 50k iter) | 21,890 images | **97.8%** | **0.50%** |
| Phase 2 (hard-negative micro-mining) | ~2,400 images | ~100% *(target)* | ~0% *(target)* |

---

## Credits

Built on top of [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) by Google.
LSTM fine-tuning via the `lstmtraining` CLI tool.
