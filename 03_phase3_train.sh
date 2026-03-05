#!/usr/bin/env bash
# =============================================================================
# Phase 3: Two-Phase LSTM Training
# =============================================================================
# Phase 1 — Initial Adaptation (Adam, LR=5e-4, append_index -1)
#   Surgically replaces only the output classification head.
#   Trains until 1% CER or 10,000 iterations.
#
# Phase 2 — Fine Refinement (Adam, LR=1e-5, hard-negative mining)
#   Full fine-tuning of entire network from Phase 1 checkpoint.
#   Trains until 0.5% CER or 20,000 iterations.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/model_output_phase3"
TRAIN_LIST="$OUTPUT_DIR/train.list"
EVAL_LIST="$OUTPUT_DIR/eval.list"
STARTER_DATA="$OUTPUT_DIR/captcha/captcha.traineddata"
OLD_TRAINEDDATA="$SCRIPT_DIR/model_output/eng_best.traineddata"
BASE_LSTM="$OUTPUT_DIR/eng_float.lstm"

echo ""
echo "============================================================"
echo "  Phase 3: Two-Phase LSTM Training"
echo "============================================================"

# ─── Sanity checks ───────────────────────────────────────────────────────────
echo ""
echo "[3.0] Pre-flight checks..."
[ -f "$TRAIN_LIST" ]  || { echo "ERROR: $TRAIN_LIST not found — run Phase 2 first"; exit 1; }
[ -f "$EVAL_LIST"  ]  || { echo "ERROR: $EVAL_LIST not found — run Phase 2 first"; exit 1; }
[ -f "$STARTER_DATA" ]|| { echo "ERROR: $STARTER_DATA not found — run Phase 2 first"; exit 1; }
[ -f "$BASE_LSTM" ]   || { echo "ERROR: $BASE_LSTM not found — run download step first"; exit 1; }
[ -f "$OLD_TRAINEDDATA" ] || { echo "ERROR: $OLD_TRAINEDDATA not found — run download step first"; exit 1; }

TRAIN_COUNT=$(wc -l < "$TRAIN_LIST")
EVAL_COUNT=$(wc -l < "$EVAL_LIST")
echo "  Train samples : $TRAIN_COUNT"
echo "  Eval  samples : $EVAL_COUNT"
echo "  Starter model : $STARTER_DATA ✓"
echo "  Base LSTM     : $BASE_LSTM ✓"

# ─── Phase 3-A: Initial Adaptation ──────────────────────────────────────────
echo ""
echo "------------------------------------------------------------"
echo "  Phase 3-A: Initial Adaptation"
echo "  Adam (net_mode=128), LR=0.0005, append_index=-1"
echo "  Max iterations: 10,000  |  Target CER: 1%"
echo "------------------------------------------------------------"
echo ""

# --traineddata points to the NEW captcha traineddata (contains target 53-class unicharset).
# append_index -1 cuts only the output head from eng_float.lstm and grafts a new 53-class head.
lstmtraining \
    --net_mode            128 \
    --model_output        "$OUTPUT_DIR/phase1_base" \
    --continue_from       "$BASE_LSTM" \
    --traineddata         "$STARTER_DATA" \
    --old_traineddata     "$OLD_TRAINEDDATA" \
    --append_index        -1 \
    --train_listfile      "$TRAIN_LIST" \
    --eval_listfile       "$EVAL_LIST" \
    --learning_rate       0.0005 \
    --target_error_rate   0.01 \
    --perfect_sample_delay 0 \
    --max_iterations      10000

PHASE1_CHECKPOINT="$OUTPUT_DIR/phase1_base_checkpoint"
echo ""
echo "[3-A] Phase 1 training complete."
echo "  Checkpoint: $PHASE1_CHECKPOINT"
ls -lh "$PHASE1_CHECKPOINT" 2>/dev/null || echo "  (checkpoint file may have a different extension)"

# ─── Phase 3-B: Fine Refinement ─────────────────────────────────────────────
echo ""
echo "------------------------------------------------------------"
echo "  Phase 3-B: Fine Refinement (Hard-Negative Mining)"
echo "  Adam (net_mode=128), LR=0.00001, delay=5"
echo "  Max iterations: 20,000  |  Target CER: 0.5%"
echo "------------------------------------------------------------"
echo ""

# Find the actual checkpoint (Tesseract may add suffixes)
if [ ! -f "$PHASE1_CHECKPOINT" ]; then
    ACTUAL_CKPT=$(ls "$OUTPUT_DIR"/phase1_base* 2>/dev/null | grep -v "\.lstm$" | tail -1 || true)
    if [ -z "$ACTUAL_CKPT" ]; then
        echo "ERROR: Could not find Phase 1 checkpoint in $OUTPUT_DIR"
        ls "$OUTPUT_DIR"/ || true
        exit 1
    fi
    echo "  Using checkpoint: $ACTUAL_CKPT"
    PHASE1_CHECKPOINT="$ACTUAL_CKPT"
fi

lstmtraining \
    --net_mode            128 \
    --model_output        "$OUTPUT_DIR/phase2_tuned" \
    --continue_from       "$PHASE1_CHECKPOINT" \
    --traineddata         "$STARTER_DATA" \
    --train_listfile      "$TRAIN_LIST" \
    --eval_listfile       "$EVAL_LIST" \
    --learning_rate       0.00001 \
    --target_error_rate   0.005 \
    --perfect_sample_delay 5 \
    --max_iterations      20000

PHASE2_CHECKPOINT="$OUTPUT_DIR/phase2_tuned_checkpoint"
echo ""
echo "[3-B] Phase 2 training complete."
echo "  Checkpoint: $PHASE2_CHECKPOINT"
ls -lh "$PHASE2_CHECKPOINT" 2>/dev/null || true

echo ""
echo "============================================================"
echo "  Phase 3 COMPLETE — run 04_export_model.sh next"
echo "============================================================"
echo ""
