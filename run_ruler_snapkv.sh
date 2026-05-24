#!/bin/bash
# SnapKV RULER evaluation — Llama-3.1-8B-Instruct.
# Config: SnapKV (kvpress), compression_ratio=0.25 @ 16k, 0.125 @ 64k/120k
#
# Usage:
#   bash run_ruler_snapkv.sh [exp_name]
#
# Example:
#   bash run_ruler_snapkv.sh snapkv_llama3_8b

set -e

MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
EXP_NAME="${1:-snapkv_ruler}"
PROMPT_CONFIG="llama3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_SCRIPT="$SCRIPT_DIR/tests/pred_ruler_snapkv.py"

COMMON_ARGS=(
    --model_path "$MODEL_PATH"
    --prompt_config "$PROMPT_CONFIG"
    --infer_script "$INFER_SCRIPT"
    --num_samples 500
    --use_cache
)

echo "============================================================"
echo "  SnapKV RULER Evaluation"
echo "  Model:       $MODEL_PATH"
echo "  Experiment:  $EXP_NAME"
echo "============================================================"

# --- 16k | ratio=0.25 ---
echo ""
echo ">>> [1/3] Sequence length: 16384 | compression_ratio=0.25"
python "$SCRIPT_DIR/run_ruler.py" \
    "${COMMON_ARGS[@]}" \
    --exp_name "${EXP_NAME}" \
    --seq_lengths 16384 \
    --compression_ratio 0.25

# --- 64k | ratio=0.125 ---
echo ""
echo ">>> [2/3] Sequence length: 65536 | compression_ratio=0.125"
python "$SCRIPT_DIR/run_ruler.py" \
    "${COMMON_ARGS[@]}" \
    --exp_name "${EXP_NAME}" \
    --seq_lengths 65536 \
    --compression_ratio 0.125

# --- 128k capped at 120k | ratio=0.125 ---
echo ""
echo ">>> [3/3] Sequence length: 120000 (128k capped) | compression_ratio=0.125"
python "$SCRIPT_DIR/run_ruler.py" \
    "${COMMON_ARGS[@]}" \
    --exp_name "${EXP_NAME}" \
    --seq_lengths 120000 \
    --compression_ratio 0.125

echo ""
echo "============================================================"
echo "  All done. Results in: results/${EXP_NAME}/"
echo "============================================================"
