#!/usr/bin/env bash
# =============================================================================
# Phase 2: Unicharset + lstmf Generation
# =============================================================================
# Steps:
#   1. Extract custom 51-char unicharset from .gt.txt files
#   2. Extract eng.lstm weights from eng.traineddata
#   3. Build captcha starter traineddata via combine_lang_model
#   4. Generate .lstmf files for every .tif in training_data/
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$SCRIPT_DIR/training_data_phase3"
OUTPUT_DIR="$SCRIPT_DIR/model_output_phase3"
TESSDATA="/usr/share/tesseract-ocr/5/tessdata"
LANGDATA_DIR="$SCRIPT_DIR/langdata_lstm"

mkdir -p "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "  Phase 2: Unicharset + lstmf Generation"
echo "============================================================"

# ─── Check prerequisites ─────────────────────────────────────────────────────
echo ""
echo "[2.0] Checking prerequisites..."
for cmd in unicharset_extractor combine_tessdata combine_lang_model tesseract; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: $cmd not found"; exit 1; }
done
echo "  All tools found ✓"
ls "$TRAIN_DIR"/*.gt.txt >/dev/null 2>&1 || {
    echo "ERROR: No .gt.txt files in $TRAIN_DIR — run 01_prepare_data.py first"
    exit 1
}
echo "  training_data/ found ✓"

# ─── Step 1: Fetch or verify langdata_lstm ──────────────────────────────────
echo ""
echo "[2.1] Checking langdata_lstm..."
if [ ! -d "$LANGDATA_DIR" ]; then
    echo "  Cloning tesseract langdata_lstm (minimal, ~30s)..."
    git clone --quiet --depth 1 \
        https://github.com/tesseract-ocr/langdata_lstm.git \
        "$LANGDATA_DIR"
    echo "  langdata_lstm cloned ✓"
else
    echo "  langdata_lstm already present ✓"
fi

# ─── Step 2: Build custom unicharset from .gt.txt files ─────────────────────
echo ""
echo "[2.2] Extracting custom unicharset from .gt.txt files..."
echo "  (Reading directly from plain-text files to avoid WordStr contamination)"

unicharset_extractor \
    --output_unicharset "$OUTPUT_DIR/custom.unicharset" \
    --norm_mode 1 \
    "$TRAIN_DIR"/*.gt.txt

# Verify "WordStr" is NOT in the unicharset
if grep -q "WordStr" "$OUTPUT_DIR/custom.unicharset"; then
    echo "  ERROR: 'WordStr' found in unicharset — extraction method failed!"
    exit 1
fi

CHAR_COUNT=$(wc -l < "$OUTPUT_DIR/custom.unicharset")
echo "  Unicharset created: $OUTPUT_DIR/custom.unicharset"
echo "  Total character entries: $CHAR_COUNT"

# ─── Step 3: Extract eng.lstm weights ───────────────────────────────────────
echo ""
echo "[2.3] Extracting LSTM weights from eng.traineddata..."
combine_tessdata -e \
    "$TESSDATA/eng.traineddata" \
    "$OUTPUT_DIR/eng.lstm"
echo "  eng.lstm extracted ✓"

# ─── Step 4: Build starter traineddata with custom unicharset ───────────────
echo ""
echo "[2.4] Building captcha starter traineddata via combine_lang_model..."
mkdir -p "$OUTPUT_DIR/captcha"

combine_lang_model \
    --input_unicharset "$OUTPUT_DIR/custom.unicharset" \
    --script_dir "$LANGDATA_DIR" \
    --lang "captcha" \
    --output_dir "$OUTPUT_DIR" \
    --pass_through_recoder true

echo "  Starter model: $OUTPUT_DIR/captcha/captcha.traineddata ✓"
ls -lh "$OUTPUT_DIR/captcha/captcha.traineddata"

# ─── Step 5: Generate .lstmf files for all training images ──────────────────
echo ""
echo "[2.5] Generating .lstmf files from .tif images..."
echo "  (This is the slow step — one tesseract call per image)"

TESSENV="TESSDATA_PREFIX=$TESSDATA"
FAIL_COUNT=0
SUCCESS_COUNT=0

for tif_file in "$TRAIN_DIR"/*.tif; do
    # Get base name WITHOUT extension for output prefix
    base="$(basename "$tif_file" .tif)"
    out_prefix="$TRAIN_DIR/$base"

    # tesseract renders this as $out_prefix.lstmf
    if [ -f "$out_prefix.lstmf" ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    elif TESSDATA_PREFIX="$TESSDATA" tesseract \
            "$tif_file" "$out_prefix" \
            --psm 13 \
            lstm.train \
            >/dev/null 2>&1; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "  [WARN] Failed: $tif_file"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # Progress update every 200 images
    TOTAL=$((SUCCESS_COUNT + FAIL_COUNT))
    if (( TOTAL % 200 == 0 )); then
        echo "  Progress: $TOTAL / $(ls "$TRAIN_DIR"/*.tif | wc -l) done..."
    fi
done

echo ""
echo "  .lstmf generation complete"
echo "  Success: $SUCCESS_COUNT  |  Failed: $FAIL_COUNT"

# ─── Step 6: Build all_lstmf.txt ────────────────────────────────────────────
echo ""
echo "[2.6] Building all_lstmf.txt list..."
rm -f "$OUTPUT_DIR/all_lstmf.txt"
find "$TRAIN_DIR" -name "*.lstmf" | sort > "$OUTPUT_DIR/all_lstmf.txt"
LSTMF_COUNT=$(wc -l < "$OUTPUT_DIR/all_lstmf.txt")
echo "  Total .lstmf files: $LSTMF_COUNT → $OUTPUT_DIR/all_lstmf.txt"

# ─── Step 7: Create train/eval split from the actual .lstmf files ───────────
echo ""
echo "[2.7] Creating 85/15 train/eval split from actual .lstmf files..."
python3 - <<'PYEOF'
import random, pathlib

all_file = pathlib.Path("/home/ca/Projects/captcha_model/model_output_phase3/all_lstmf.txt")
lines = [l.strip() for l in all_file.read_text().splitlines() if l.strip()]
random.seed(42)
random.shuffle(lines)
split = int(len(lines) * 0.85)
train, eval_ = lines[:split], lines[split:]

out = pathlib.Path("/home/ca/Projects/captcha_model/model_output_phase3")
(out / "train.list").write_text("\n".join(train) + "\n")
(out / "eval.list").write_text("\n".join(eval_) + "\n")
print(f"  train.list: {len(train)} entries")
print(f"  eval.list : {len(eval_)} entries")
PYEOF

echo ""
echo "============================================================"
echo "  Phase 2 COMPLETE — run 03_train.sh next"
echo "============================================================"
echo ""
