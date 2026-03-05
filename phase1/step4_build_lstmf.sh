#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Running from: $SCRIPT_DIR"

OUTPUT_DIR="$SCRIPT_DIR/model_output_phase4"
DATASET_DIR="$SCRIPT_DIR/training_data_phase4"
TESSDATA_PREFIX="$(dirname "$(dirname $(which tesseract))")/share/tessdata"
# Keep TESSDATA explicitly defined just in case system tessdata is elsewhere
export TESSDATA_PREFIX="/usr/share/tesseract-ocr/4.00/tessdata"

echo "============================================================"
echo "  Phase 4: Parallel Extract Unicharset & Build .lstmf"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

# 1. Gather all Box Files
echo "[1.1] Gathering box files from $DATASET_DIR..."
BOX_FILES=()
while IFS=  read -r -d $'\0'; do
    BOX_FILES+=("$REPLY")
done < <(find "$DATASET_DIR" -maxdepth 1 -name "*.box" -print0)

NUM_BOX_FILES=${#BOX_FILES[@]}
echo "  Found $NUM_BOX_FILES .box files."
if [ "$NUM_BOX_FILES" -eq 0 ]; then
  echo "  No .box files found! Check dataset directory."
  exit 1
fi

cat /dev/null > "$OUTPUT_DIR/all_box_files.txt"
for f in "${BOX_FILES[@]}"; do
    echo "$f" >> "$OUTPUT_DIR/all_box_files.txt"
done

# 2. Unicharset Extraction
UNICHARSET_FILE="$OUTPUT_DIR/custom.unicharset"
if [ ! -f "$UNICHARSET_FILE" ]; then
    echo "[1.2] Extracting Unicharset..."
    unicharset_extractor --output_unicharset "$UNICHARSET_FILE" --norm_mode 2 "${BOX_FILES[@]}"
    echo "  -> Saved to $UNICHARSET_FILE"
else
    echo "[1.2] $UNICHARSET_FILE already exists, skipping unicharset extraction."
fi

# 3. Combine Lang Model (Unicharset + Dawgs)
STARTER_DATA_DIR="$OUTPUT_DIR/captcha"
mkdir -p "$STARTER_DATA_DIR"
mkdir -p "$SCRIPT_DIR/langdata_lstm/captcha"
touch "$SCRIPT_DIR/langdata_lstm/captcha/captcha.config"

echo "[1.3] Combining Language Model..."
if [ ! -f "$STARTER_DATA_DIR/captcha.traineddata" ]; then
    combine_lang_model \
      --input_unicharset "$UNICHARSET_FILE" \
      --script_dir "$SCRIPT_DIR/langdata_lstm" \
      --output_dir "$STARTER_DATA_DIR" \
      --lang captcha \
      --pass_through_recoder
else
    echo "  -> $STARTER_DATA_DIR/captcha.traineddata already generated"
fi

# 4. Generate .lstmf Parallelly
echo "[1.4] Generating .lstmf files (Parallel xargs)..."

TIF_FILES=()
while IFS=  read -r -d $'\0'; do
    TIF_FILES+=("$REPLY")
done < <(find "$DATASET_DIR" -maxdepth 1 -name "*.tif" -print0)
NUM_TIFS=${#TIF_FILES[@]}
echo "  Total .tif images: $NUM_TIFS"

cat /dev/null > "$OUTPUT_DIR/tif_list.txt"
for f in "${TIF_FILES[@]}"; do
    echo "$f" >> "$OUTPUT_DIR/tif_list.txt"
done

# We create a dummy loop script that xargs runs across cores
cat << 'EOF' > "$OUTPUT_DIR/do_lstmf.sh"
#!/bin/bash
TIF_FILE="$1"
BASE_NAME="$(basename "${TIF_FILE%.*}")"
OUTPUT_DIR="$2"
# Check if already done
if [ -f "${TIF_FILE%.*}.lstmf" ]; then
    exit 0
fi
tesseract "$TIF_FILE" "${TIF_FILE%.*}" --psm 13 lstm.train 2>/dev/null
EOF
chmod +x "$OUTPUT_DIR/do_lstmf.sh"

echo "  Running 'tesseract ... lstm.train' heavily parallelized... (this is fast!)"
cat "$OUTPUT_DIR/tif_list.txt" | xargs -n 1 -P $(nproc) -I {} bash "$OUTPUT_DIR/do_lstmf.sh" "{}" "$OUTPUT_DIR"

echo "  .lstmf generation complete."

# 5. Build Train/Eval List 
echo "[1.5] Building train/eval split..."
LSTMF_FILES=()
while IFS=  read -r -d $'\0'; do
    LSTMF_FILES+=("$REPLY")
done < <(find "$DATASET_DIR" -maxdepth 1 -name "*.lstmf" -print0)

TOTAL_LSTMF=${#LSTMF_FILES[@]}
echo "  Total valid .lstmf files: $TOTAL_LSTMF"

cat /dev/null > "$OUTPUT_DIR/all_lstmf.txt"
for f in "${LSTMF_FILES[@]}"; do
    echo "$f" >> "$OUTPUT_DIR/all_lstmf.txt"
done

NUM_TRAIN=$(echo "$TOTAL_LSTMF * 0.85" | bc | cut -d'.' -f1)
NUM_EVAL=$((TOTAL_LSTMF - NUM_TRAIN))

shuf "$OUTPUT_DIR/all_lstmf.txt" > "$OUTPUT_DIR/shuffled.txt"
head -n "$NUM_TRAIN" "$OUTPUT_DIR/shuffled.txt" > "$OUTPUT_DIR/train.list"
tail -n "$NUM_EVAL" "$OUTPUT_DIR/shuffled.txt" > "$OUTPUT_DIR/eval.list"

echo "  train.list: $(wc -l < "$OUTPUT_DIR/train.list") entries"
echo "  eval.list : $(wc -l < "$OUTPUT_DIR/eval.list") entries"

echo "============================================================"
echo "  Phase 4 Prep COMPLETE — run 03_phase4_train.sh next"
echo "============================================================"
