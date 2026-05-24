# LASER-KV

**LASER-KV** enables infinite-context reasoning within a fixed memory budget by iteratively compressing the KV cache. It is built on **BMRT** (Bounded-Memory Recursive Transformer) — a modular compression engine with pluggable selection strategies.

**LASER-KV** is specifically defined as the `hybrid(exact + magicpig_baseline)` strategy at `ratio=0.75` in `accumulate` mode. This combines exact attention scoring (75% of the global budget) with LSH-based MagicPIG sampling (25%) to balance precision and diversity.

## Features

- **Recursive Compression**: `State_N = Compress(State_{N-1} + Chunk_N)` — unbounded context in fixed memory.
- **Accumulate Mode**: Keeps previous local tail separately; only merges when needed. Defined as LASER-KV when paired with exact+magicpig hybrid at 0.75 ratio.
- **Dynamic Budgeting**: Budget is split via `protection_divisor` into Anchors, Local Window, and Global Memory.
- **Three Selection Strategies**:
  1. **Exact Attention** — true top-K by summed softmax scores across all layers.
  2. **LSH Sampling** — hash collision frequency (`frequency_rank`, ours) or probabilistic buckets (`magicpig_baseline`).
  3. **Hybrid** — primary + secondary selector split, e.g. Exact + LSH.
- **Flash Attention Kernel**: A post-hoc Q-recomputation kernel (`flash_scorer.py`) and position-ID patch (`patch_fa.py`) unlock Flash Attention for **all** strategies including Exact and Hybrid — not just LSH.

## Architecture

### Budget Allocation

Total Budget $B$ is split based on `protection_divisor` ($n$):

| Slot | Size | Role |
| :--- | :--- | :--- |
| **Anchors** | $B/n$ | Fixed initial tokens — stabilise attention |
| **Local Window** | $B/n$ | Most recent tokens — preserve coherence |
| **Global Memory** | $B - 2B/n$ | Tokens selected by the configured strategy |

### Compression Modes

| Mode | Rescoring Scope | Description |
| :--- | :--- | :--- |
| **Accumulate** | `prev_tail + current_block` | Previous local tail kept separately; efficient incremental updates |
| **Recursive** | Entire history | All non-anchor tokens rescored each step; tokens can be "resurrected" |

### Flash Attention Kernel

Two components make Flash Attention compatible with exact scoring:

- **`bmrt/flash_scorer.py`** — captures hidden states via forward pre-hooks, recomputes Q post-hoc (with RoPE), then computes `softmax(Q @ K^T / sqrt(d))` per layer. Produces scores identical to the eager accumulator without materialising the attention weight matrix.
- **`bmrt/patch_fa.py`** — patches `transformers.modeling_flash_attention_utils` to handle disjoint `position_ids` in BMRT's compressed sequences (single-sequence varlen path). Applied as a context manager during generation.

This means `--backend flash` works correctly with `--method exact` and `--method hybrid`, not just `--method lsh`.

### Components

| Module | Role |
| :--- | :--- |
| `bmrt.processor` | `RecursiveCompressionEngine` — main loop and state management |
| `bmrt.selectors.eager_exact` | Exact attention scoring (eager accumulator or flash scorer) |
| `bmrt.selectors.lsh_core` | LSH selection: `frequency_rank` (ours) or `magicpig_baseline` |
| `bmrt.selectors.hybrid` | Combines two selectors with a configurable budget split ratio |
| `bmrt.accumulator` | Hook-based score accumulation for eager backend |
| `bmrt.flash_scorer` | Post-hoc Q-recomputation kernel for Flash Attention backend |
| `bmrt.patch_fa` | Position-ID patch enabling varlen Flash Attention on compressed sequences |

## Installation

```bash
pip install -e .
```

**Dependencies**:
- `torch >= 2.2.0`
- `transformers`
- `datasets`
- `accelerate`
- `flash-attn` (required for `--backend flash`)

## Usage

### LASER-KV (Recommended)

The canonical LASER-KV configuration: hybrid exact+magicpig at 0.75 ratio, accumulate mode, flash backend.

```bash
python run_single_sample.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --method hybrid \
    --hybrid_primary exact \
    --hybrid_secondary lsh \
    --lsh_mode magicpig_baseline \
    --hybrid_ratio 0.75 \
    --compression_mode accumulate \
    --backend flash \
    --compression_ratio 0.75 \
    --budget 4096
```

### 1. Exact Strategy (Flash or Eager)

Ground-truth selection via summed softmax scores. Supports both backends.

```bash
# Flash backend (recommended — faster, same quality)
python run_single_sample.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --method exact \
    --backend flash \
    --budget 4096 \
    --protection_divisor 4

# Eager backend
python run_single_sample.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --method exact \
    --backend eager \
    --budget 4096 \
    --protection_divisor 4
```

### 2. LSH Strategy (Flash Attention)

Approximate selection via hash collision frequency.

```bash
python run_single_sample.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --method lsh \
    --lsh_mode frequency_rank \
    --backend flash \
    --budget 4096 \
    --protection_divisor 4
```

### 3. LSH Baseline (MagicPIG Style)

Probabilistic bucket sampling from LSH tables.

```bash
python run_single_sample.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --method lsh \
    --lsh_mode magicpig_baseline \
    --backend flash \
    --budget 4096
```

### 4. Hybrid Strategy (Flash or Eager)

Combines two selectors. The flash scorer enables Flash Attention even when Exact is the primary.

```bash
python run_single_sample.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --method hybrid \
    --hybrid_primary exact \
    --hybrid_secondary lsh \
    --lsh_mode magicpig_baseline \
    --hybrid_ratio 0.75 \
    --compression_mode accumulate \
    --backend flash \
    --budget 4096
```

## Ablation Grid

Recommended runs for systematic evaluation (~14 core configs per dataset):

| Strategy | Compression Mode | Backend | Ratio | Note |
| :--- | :--- | :--- | :--- | :--- |
| Exact | Accumulate | Flash | — | Gold standard |
| Exact | Recursive | Flash | — | Full history rescoring |
| LSH `frequency_rank` | Accumulate | Flash | — | Our LSH method |
| LSH `frequency_rank` | Recursive | Flash | — | |
| LSH `magicpig_baseline` | Accumulate | Flash | — | MagicPIG baseline |
| LSH `magicpig_baseline` | Recursive | Flash | — | |
| Hybrid (Exact + `magicpig`) | Accumulate | Flash | 0.50 | |
| Hybrid (Exact + `magicpig`) | Accumulate | Flash | 0.75 | **LASER-KV** |
| Hybrid (Exact + `magicpig`) | Recursive | Flash | 0.50 | |
| Hybrid (Exact + `magicpig`) | Recursive | Flash | 0.75 | |
| Hybrid (Exact + `freq_rank`) | Accumulate | Flash | 0.50 | |
| Hybrid (Exact + `freq_rank`) | Accumulate | Flash | 0.75 | |
| Hybrid (Exact + `freq_rank`) | Recursive | Flash | 0.50 | |
| Hybrid (Exact + `freq_rank`) | Recursive | Flash | 0.75 | |

Flash backend is supported for all strategies via the post-hoc scoring kernel.

## Python API

```python
from bmrt import RecursiveCompressionEngine

# LASER-KV configuration
engine = RecursiveCompressionEngine(
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    selector_type='hybrid',
    hybrid_primary='exact',
    hybrid_secondary='lsh',
    lsh_mode='magicpig_baseline',
    hybrid_ratio=0.75,
    compression_mode='accumulate',
    backend='flash',
    budget=4096,
    protection_divisor=4,
)

result = engine(
    prompt_context="<your long context here>",
    prompt_query="What is the main topic?"
)

print(result['text'][0])
```

## Evaluation

### BabiLong (Synthetic QA)

```bash
python run_single_sample.py \
    --model_path gradientai/Llama-3-8B-Instruct-Gradient-1048k \
    --method hybrid \
    --hybrid_primary exact \
    --hybrid_secondary lsh \
    --lsh_mode magicpig_baseline \
    --hybrid_ratio 0.75 \
    --compression_mode accumulate \
    --backend flash \
    --compression_ratio 0.75 \
    --dataset_config 128k \
    --dataset_split qa1 \
    --num_samples 100 \
    --results_file accuracies_laser_kv.txt
```

### LongBench v2

```bash
# Run inference
python tests/pred.py \
    --model_path gradientai/Llama-3-8B-Instruct-Gradient-1048k \
    --method hybrid \
    --compression_ratio 0.75 \
    --block_size 4096 \
    --save_dir results

# Evaluate with official script
python eval.py --model <model_name>
```
