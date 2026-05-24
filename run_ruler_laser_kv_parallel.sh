#!/bin/bash
# LASER-KV RULER parallel evaluation across 2 GPUs.
#
# GPU 0: 16k (ratio=0.25) then 64k (ratio=0.125)  -- sequential on GPU 0
# GPU 1: 120k (ratio=0.125, 128k capped)           -- dedicated GPU 1
#
# Usage:
#   bash run_ruler_laser_kv_parallel.sh [exp_name]

set -e

MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
EXP_NAME="${1:-laser_kv_ruler}"
PROMPT_CONFIG="llama3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMON_ARGS=(
    --model_path "$MODEL_PATH"
    --exp_name "$EXP_NAME"
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
    --num_samples 100
    --use_cache
)

echo "============================================================"
echo "  LASER-KV RULER Parallel Evaluation (2x A100)"
echo "  GPU 0: 16k + 64k"
echo "  GPU 1: 120k (128k capped)"
echo "============================================================"

# GPU 0: 16k then 64k (sequential within GPU 0)
CUDA_VISIBLE_DEVICES=0 python3 "$SCRIPT_DIR/run_ruler.py" \
    "${COMMON_ARGS[@]}" \
    --seq_lengths 16384 \
    --compression_ratio 0.25 \
    &
GPU0_PID=$!
echo "GPU 0 started (PID $GPU0_PID): 16k"

# GPU 1: 120k dedicated
CUDA_VISIBLE_DEVICES=1 python3 "$SCRIPT_DIR/run_ruler.py" \
    "${COMMON_ARGS[@]}" \
    --seq_lengths 120000 \
    --compression_ratio 0.125 \
    &
GPU1_PID=$!
echo "GPU 1 started (PID $GPU1_PID): 120k"

# Wait for 16k to finish, then run 64k on GPU 0
wait $GPU0_PID
echo ""
echo ">>> GPU 0: 16k done. Starting 64k..."

CUDA_VISIBLE_DEVICES=0 python3 "$SCRIPT_DIR/run_ruler.py" \
    "${COMMON_ARGS[@]}" \
    --seq_lengths 65536 \
    --compression_ratio 0.125 \
    &
GPU0_64K_PID=$!
echo "GPU 0 started (PID $GPU0_64K_PID): 64k"

# Wait for both remaining jobs
wait $GPU0_64K_PID
echo ">>> GPU 0: 64k done."

wait $GPU1_PID
echo ">>> GPU 1: 120k done."

echo ""
echo "============================================================"
echo "  All done. Results in: results/${EXP_NAME}/"
echo "============================================================"
