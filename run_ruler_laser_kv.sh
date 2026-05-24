#!/bin/bash
# LASER-KV RULER evaluation — Llama-3.1-8B-Instruct.
# Config: hybrid(exact+magicpig, ratio=0.75), flash, accumulate, block=4096, bits=10, tables=8
#
# Usage:
#   bash run_ruler_laser_kv.sh [exp_name]
#
# Example:
#   bash run_ruler_laser_kv.sh laser_kv_llama3_8b

set -e

MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
EXP_NAME="${1:-laser_kv_ruler}"
PROMPT_CONFIG="llama3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMON_ARGS=(
    --model_path "$MODEL_PATH"
    --prompt_config "$PROMPT_CONFIG"
    --method hybrid
    --backend flash
    --compression_mode accumulate
    --hybrid_primary exact
    --hybrid_secondary lsh
    --lsh_mode magicpig_baseline
    --hybrid_ratio 0.75
    --block_size 4096
    --num_bits 10
    --num_tables 8
    --num_samples 500
    --use_cache
)

echo "============================================================"
echo "  LASER-KV RULER Evaluation"
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
