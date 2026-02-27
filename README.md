# Tesseract 5 Captcha Fine-Tuning Pipeline

This repository contains a full, production-ready pipeline for fine-tuning a Tesseract 5 LSTM model to recognize highly distorted, noisy web captchas. Through a combination of specific LSTM topological adjustments (surgical fine-tuning) and heavy data augmentation, the final Int8 quantized model achieves **97.5% exact-match accuracy** and a **0.47% Character Error Rate (CER)**.

## Overview of the Pipeline

The pipeline is split into 5 distinct phases represented by numerical scripts in the root directory:

1. **`01_prepare_data.py`**:
   Ingests the raw `png` images, applies Otsu binarization, pads the 203x50 images to 253x100 (for LSTM receptive fields), and generates `.tif` images, `.gt.txt` ground truth labels, and Tesseract `.box` files (specifically adapting the `WordStr` bounding box syntax for Tesseract 5).

2. **`01b_augment_data.py`** (The Augmentation Engine):
   Massively expands the baseline dataset (e.g. from 1.8k to 10.9k images) by applying four transformations to each training image:
   - Rotation (±3 degrees)
   - Elastic/Wave Distortion (simulates ripples)
   - Morphological Distortion (slight erosion/dilation)
   - Additive Gaussian Noise (injects static)

3. **`02_build_unicharset_and_lstmf.sh`**:
   Extracts testing/training parameters. Builds a specialized minimum-subset `unicharset` (54 characters total) to restrict the model from hallucinating outside the captcha scope. Compiles the image/box structures into Tesseract multidimensional `.lstmf` training tensors.

4. **`03_train.sh`** (The Training Engine):
   A two-phase training loop on top of `eng_best.traineddata`.
   * **Phase 3-A**: Uses `--append_index -1` to cut off the English softmax layer and snap on the 54-class captcha unicharset head. Uses the Adam optimizer (`net_mode=128`) and trains extensively (10k+ iterations).
   * **Phase 3-B**: Restarts with a lower learning rate. Introduces `--perfect_sample_delay 5` which triggers **Hard-Negative Mining**. The network isolates its training specifically on the hardest, most augmented edge-cases until the CER drops below 1%.

5. **`04_export_model.sh`**:
   Strips the massive Adam optimizer momentum states out of the `.checkpoint` files. Employs `--convert_to_int` to quantize the 32-bit float model down into an 8-bit integer sequence. The result is a lightning-fast, production-ready 1.4MB `.traineddata` file.

6. **`05_test_inference.py`**:
   A standalone validation script that pulls unseen files randomly from an `eval.list`, applies the identical preprocessing CV2 pipeline, runs `pytesseract` using the new model (with `--psm 13`), and calculates the final Exact Match & CER metrics across the test pool.

## Requirements & Installation

You need Tesseract 5 and standard Python ML/CV libraries installed.
First, clone the repository and install the python dependencies:

```bash
git clone https://github.com/Shreyxpatil/tesseract5-captcha-finetuning.git
cd tesseract5-captcha-finetuning
pip install -r requirements.txt
```

You must have **Tesseract 5** and its training tools installed on your operating system.

**For Ubuntu / Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng libtesseract-dev libleptonica-dev
sudo apt install tesseract-ocr-script-latn
# Install the training tools (crucial for lstmtraining, combine_tessdata, etc)
sudo apt install tesseract-ocr-all
```

## How to Run the Pipeline

Place your labeled `.png` captcha files inside a `training_data/` directory (where the filename is the label, e.g., `aB3dE.png`).

Run the scripts in sequential order:

```bash
python3 01_prepare_data.py
python3 01b_augment_data.py
bash 02_build_unicharset_and_lstmf.sh
bash 03_train.sh
bash 04_export_model.sh
python3 05_test_inference.py
```

### Important Notes on the Training Run
- Phase 3 will download the baseline floating-point `eng.traineddata` (from `tessdata_best` on GitHub). Standard OS-installed Tesseract models are `int8` (fast integer models) and *cannot* be retrained.
- Training generates `checkpoint` files in a massive size. When running `04_export_model.sh` at the end, the system automatically quantizes them down to your production output size (~1.4MB).

## Pretrained Model
The final, highly tuned model from our run is located at:
`model_output/captcha_final.traineddata`

**To use it in your code:**
```python
import pytesseract
import cv2

# Ensure your model is in the current directory or provide the tessdata-dir path
config = '--tessdata-dir model_output -l captcha_final --psm 13'

image = cv2.imread('test_captcha.png')
# (Apply binarization/padding preprocessing here)
text = pytesseract.image_to_string(image, config=config)
print(f"Prediction: {text}")
```
