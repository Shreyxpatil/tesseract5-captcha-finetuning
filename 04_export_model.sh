#!/usr/bin/env bash
# =============================================================================
# Phase 4: Model Export + Int8 Quantization
# =============================================================================
# Converts Phase 2 checkpoint to production-ready traineddata:
#   --stop_training    : strips optimizer state (Adam momentums etc.)
#   --convert_to_int   : quantizes fp32 weights → int8 (~75% size reduction)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/model_output"
STARTER_DATA="$OUTPUT_DIR/captcha/captcha.traineddata"
PHASE2_CHECKPOINT="$OUTPUT_DIR/phase2_tuned_checkpoint"
FINAL_MODEL="$OUTPUT_DIR/captcha_final.traineddata"

echo ""
echo "============================================================"
echo "  Phase 4: Export + Int8 Quantization"
echo "============================================================"
echo ""

# ─── Find the Phase 2 checkpoint ─────────────────────────────────────────────
if [ ! -f "$PHASE2_CHECKPOINT" ]; then
    echo "[4.0] Standard checkpoint path not found, searching..."
    PHASE2_CHECKPOINT=$(ls "$OUTPUT_DIR"/phase2_tuned* 2>/dev/null | grep -v "\.lstm$" | tail -1 || true)
    if [ -z "$PHASE2_CHECKPOINT" ]; then
        echo "ERROR: Phase 2 checkpoint not found in $OUTPUT_DIR"
        echo "  Available files:"
        ls "$OUTPUT_DIR"/ 2>/dev/null || true
        exit 1
    fi
fi
echo "  Checkpoint : $PHASE2_CHECKPOINT"
ls -lh "$PHASE2_CHECKPOINT"

# ─── Export model ─────────────────────────────────────────────────────────────
echo ""
echo "[4.1] Exporting int8-quantized model..."
lstmtraining \
    --stop_training \
    --convert_to_int \
    --continue_from  "$PHASE2_CHECKPOINT" \
    --traineddata    "$STARTER_DATA" \
    --model_output   "$FINAL_MODEL"

echo ""
if [ -f "$FINAL_MODEL" ]; then
    SIZE=$(du -sh "$FINAL_MODEL" | cut -f1)
    echo "  ✅ Production model saved: $FINAL_MODEL"
    echo "  📦 Size (int8): $SIZE"
else
    echo "  ❌ Export failed — $FINAL_MODEL not found"
    exit 1
fi

echo ""
echo "============================================================"
echo "  Phase 4 COMPLETE"
echo ""
echo "  To use in production:"
echo "  TESSDATA_PREFIX=$OUTPUT_DIR tesseract <image> stdout \\"
echo "    --tessdata-dir $OUTPUT_DIR \\"
echo "    -l captcha_final --psm 13"
echo "============================================================"
echo ""
