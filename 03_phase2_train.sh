#!/usr/bin/env bash
# =============================================================================
# Phase 3: Train on Hard Negatives
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/model_output_phase2"
TRAIN_LIST="$OUTPUT_DIR/train.list"
EVAL_LIST="$OUTPUT_DIR/eval.list"
STARTER_DATA="$OUTPUT_DIR/captcha/captcha.traineddata"
OLD_TRAINEDDATA="$SCRIPT_DIR/model_output/captcha_final.traineddata"
BASE_LSTM="$OUTPUT_DIR/captcha_final.lstm"

echo ""
echo "============================================================"
echo "  Phase 3: Train on Hard Negatives"
echo "============================================================"

[ -f "$TRAIN_LIST" ]  || { echo "ERROR: $TRAIN_LIST not found"; exit 1; }
[ -f "$OLD_TRAINEDDATA" ] || { echo "ERROR: $OLD_TRAINEDDATA not found"; exit 1; }

echo "[3.1] Extracting LSTM weights from captcha_final.traineddata..."
combine_tessdata -e "$OLD_TRAINEDDATA" "$BASE_LSTM"

echo ""
echo "------------------------------------------------------------"
echo "  Fine Refinement (Hard-Negative Mining)"
echo "  Adam (net_mode=128), LR=0.00001, delay=5"
echo "  Max iterations: 10,000  |  Target CER: 0.1%"
echo "------------------------------------------------------------"

lstmtraining \
    --net_mode            128 \
    --model_output        "$OUTPUT_DIR/phase2_tuned" \
    --continue_from       "$BASE_LSTM" \
    --traineddata         "$STARTER_DATA" \
    --train_listfile      "$TRAIN_LIST" \
    --eval_listfile       "$EVAL_LIST" \
    --learning_rate       0.00001 \
    --target_error_rate   0.001 \
    --perfect_sample_delay 5 \
    --max_iterations      10000

PHASE2_CHECKPOINT="$OUTPUT_DIR/phase2_tuned_checkpoint"
echo ""
echo "[3] Training complete."
echo "  Checkpoint: $PHASE2_CHECKPOINT"

echo ""
echo "============================================================"
echo "  COMPLETE — run export script next"
echo "============================================================"
