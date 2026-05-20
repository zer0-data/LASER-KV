#!/usr/bin/env bash
# Babilong evaluation: SnapKV, PyramidKV (16k/64k), and BMRT-LSH magicpig_baseline (16k)
# Model: meta-llama/Llama-3.1-8B-Instruct
# Tasks: qa4 qa7 qa8 qa9 qa10

set -euo pipefail

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PY=${PYTHON:-python}
TASKS=(qa4 qa7 qa8 qa9 qa10)

SNAPKV_RESULTS="accuracies_snapkv.txt"
PYRAMIDKV_RESULTS="accuracies_pyramidkv.txt"
BMRT_RESULTS="accuracies_bmrt_lsh.txt"

run() {
    echo ""
    echo "+ $*"
    "$@" || echo "[WARN] Command failed, continuing..."
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SnapKV + PyramidKV @ 16k, compression_ratio=0.25
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " SECTION 1: SnapKV + PyramidKV  |  16k  |  ratio=0.25"
echo "================================================================"

for task in "${TASKS[@]}"; do
    echo ""
    echo "--- SnapKV | 16k | $task ---"
    run $PY tests/babilong_snapkv.py \
        --model_path "$MODEL" \
        --dataset_config 16k \
        --dataset_split "$task" \
        --compression_ratio 0.25 \
        --results_file "$SNAPKV_RESULTS"
done

for task in "${TASKS[@]}"; do
    echo ""
    echo "--- PyramidKV | 16k | $task ---"
    run $PY tests/babilong_pyramidkv.py \
        --model_path "$MODEL" \
        --dataset_config 16k \
        --dataset_split "$task" \
        --compression_ratio 0.25 \
        --results_file "$PYRAMIDKV_RESULTS"
done

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — SnapKV + PyramidKV @ 64k, compression_ratio=0.125
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " SECTION 2: SnapKV + PyramidKV  |  64k  |  ratio=0.125"
echo "================================================================"

for task in "${TASKS[@]}"; do
    echo ""
    echo "--- SnapKV | 64k | $task ---"
    run $PY tests/babilong_snapkv.py \
        --model_path "$MODEL" \
        --dataset_config 64k \
        --dataset_split "$task" \
        --compression_ratio 0.125 \
        --results_file "$SNAPKV_RESULTS"
done

for task in "${TASKS[@]}"; do
    echo ""
    echo "--- PyramidKV | 64k | $task ---"
    run $PY tests/babilong_pyramidkv.py \
        --model_path "$MODEL" \
        --dataset_config 64k \
        --dataset_split "$task" \
        --compression_ratio 0.125 \
        --results_file "$PYRAMIDKV_RESULTS"
done

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — BMRT LSH (magicpig_baseline) @ 16k, compression_ratio=0.25
#   3a: num_bits=10, num_tables=8
#   3b: num_bits=8,  num_tables=100
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " SECTION 3a: BMRT LSH magicpig_baseline  |  16k  |  bits=10  tables=8"
echo "================================================================"

for task in "${TASKS[@]}"; do
    echo ""
    echo "--- BMRT LSH magicpig | 16k | bits=10 tables=8 | $task ---"
    run $PY run_single_sample.py \
        --model_path "$MODEL" \
        --method lsh \
        --lsh_mode magicpig_baseline \
        --backend flash \
        --compression_mode accumulate \
        --compression_ratio 0.25 \
        --block_size 4096 \
        --num_bits 10 \
        --num_tables 8 \
        --dataset_config 16k \
        --dataset_split "$task" \
        --results_file "$BMRT_RESULTS"
done

echo ""
echo "================================================================"
echo " SECTION 3b: BMRT LSH magicpig_baseline  |  16k  |  bits=8  tables=100"
echo "================================================================"

for task in "${TASKS[@]}"; do
    echo ""
    echo "--- BMRT LSH magicpig | 16k | bits=8 tables=100 | $task ---"
    run $PY run_single_sample.py \
        --model_path "$MODEL" \
        --method lsh \
        --lsh_mode magicpig_baseline \
        --backend flash \
        --compression_mode accumulate \
        --compression_ratio 0.25 \
        --block_size 4096 \
        --num_bits 8 \
        --num_tables 100 \
        --dataset_config 16k \
        --dataset_split "$task" \
        --results_file "$BMRT_RESULTS"
done

echo ""
echo "================================================================"
echo " All runs complete."
echo " Results appended to:"
echo "   $SNAPKV_RESULTS"
echo "   $PYRAMIDKV_RESULTS"
echo "   $BMRT_RESULTS"
echo "================================================================"
