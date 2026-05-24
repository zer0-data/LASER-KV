"""
BMRT RULER benchmark orchestrator.

Chains data generation -> BMRT inference -> evaluation for all
sequence lengths and tasks, then aggregates results.

Usage (LASER-KV defaults):
    python run_ruler.py \
        --model_path meta-llama/Llama-3.1-8B-Instruct \
        --exp_name laser_kv_llama3_8b \
        --seq_lengths 4096 8192 16384 32768 \
        --prompt_config llama3

Pre-generated data:
    python run_ruler.py \
        --model_path meta-llama/Llama-3.1-8B-Instruct \
        --exp_name laser_kv_llama3_8b \
        --seq_lengths 16384 32768 \
        --data_dir /path/to/existing/data \
        --use_cache

Custom method:
    python run_ruler.py \
        --model_path ... \
        --exp_name exact_accumulate \
        --method exact \
        --backend flash \
        --compression_mode accumulate \
        --compression_ratio 0.75
"""
import argparse
import os
import subprocess
import sys

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RULER_DIR = os.path.join(CURRENT_DIR, 'ruler')
DATA_SCRIPT = os.path.join(RULER_DIR, 'data', 'prepare.py')
DEFAULT_INFER_SCRIPT = os.path.join(CURRENT_DIR, 'tests', 'pred_ruler.py')
EVAL_SCRIPT = os.path.join(RULER_DIR, 'eval', 'evaluate.py')
GATHER_SCRIPT = os.path.join(RULER_DIR, 'gather_results_ruler.py')

with open(os.path.join(RULER_DIR, 'synthetic_task_config.yaml')) as f:
    TASK_CONFIG = yaml.safe_load(f)
with open(os.path.join(RULER_DIR, 'synthetic_inference_config.yaml')) as f:
    INFERENCE_CONFIG = yaml.safe_load(f)

ALL_TASKS = list(TASK_CONFIG.keys())


def run(cmd):
    print(f'\n$ {cmd}\n')
    subprocess.run(cmd, shell=True, check=True, text=True)


def main(args):
    tasks = args.tasks if args.tasks else ALL_TASKS
    output_base = os.path.join(CURRENT_DIR, 'results', args.exp_name)
    infer_script = args.infer_script or DEFAULT_INFER_SCRIPT

    stop_words_flag = f"--stop_words '{args.stop_words}'" if args.stop_words else ''
    window_size_flag = f'--window_size {args.window_size}' if args.window_size else ''

    for seq_len in args.seq_lengths:
        print(f'\n{"="*60}')
        print(f'  Sequence Length: {seq_len}')
        print(f'{"="*60}')

        results_dir = os.path.join(output_base, str(seq_len))
        data_dir = args.data_dir or os.path.join(output_base, 'data', str(seq_len))
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        for task in tasks:
            print(f'\n--- Task: {task} ---')
            task_data = os.path.join(data_dir, task, 'validation.jsonl')
            task_output = os.path.join(results_dir, f'{task}.jsonl')
            tokens_to_generate = INFERENCE_CONFIG['tokens_to_generate'].get(task, 128)

            # Step 1: Data generation
            if not os.path.exists(task_data) or args.regen_data:
                run(
                    f'python {DATA_SCRIPT}'
                    f' --save_dir {data_dir}'
                    f' --task {task}'
                    f' --tokenizer_path {args.model_path}'
                    f' --tokenizer_type hf'
                    f' --max_seq_length {seq_len}'
                    f' --model_template_type {args.prompt_config}'
                    f' --num_samples {args.num_samples}'
                )
            else:
                print(f'  Data exists, skipping: {task_data}')

            # Step 2: BMRT inference
            if os.path.exists(task_output) and not args.rerun_inference and not args.use_cache:
                print(f'  Output exists, skipping: {task_output}')
                continue

            run(
                f'python {infer_script}'
                f' --input_path {task_data}'
                f' --output_path {task_output}'
                f' --model_path {args.model_path}'
                f' --tokens_to_generate {tokens_to_generate}'
                f' --seq_length {seq_len}'
                f' --method {args.method}'
                f' --backend {args.backend}'
                f' --compression_mode {args.compression_mode}'
                f' --compression_ratio {args.compression_ratio}'
                f' --hybrid_primary {args.hybrid_primary}'
                f' --hybrid_secondary {args.hybrid_secondary}'
                f' --lsh_mode {args.lsh_mode}'
                f' --hybrid_ratio {args.hybrid_ratio}'
                f' --protection_divisor {args.protection_divisor}'
                f' --block_size {args.block_size}'
                f' --num_bits {args.num_bits}'
                f' --num_tables {args.num_tables}'
                + (f' {stop_words_flag}' if stop_words_flag else '')
                + (f' {window_size_flag}' if window_size_flag else '')
                + (' --use_cache' if args.use_cache else '')
            )

        # Step 3: Evaluate all tasks at this seq_len
        run(
            f'python {EVAL_SCRIPT}'
            f' --data_dir {results_dir}'
            f' --benchmark synthetic'
        )

    # Step 4: Aggregate results across all seq lengths
    run(f'python {GATHER_SCRIPT} -e {output_base}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--exp_name', default='bmrt_ruler')
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[4096, 8192, 16384, 32768])
    parser.add_argument('--tasks', type=str, nargs='*', default=None,
                        help='Subset of tasks; default: all 13')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--prompt_config', default='llama3', choices=['base', 'llama3', 'qwen3'])
    parser.add_argument('--data_dir', default=None,
                        help='Pre-generated data dir; skips data generation if set')
    parser.add_argument('--regen_data', action='store_true',
                        help='Force re-generation even if data exists')
    parser.add_argument('--rerun_inference', action='store_true',
                        help='Force re-run inference even if output exists')
    parser.add_argument('--use_cache', action='store_true',
                        help='Resume inference from existing output (append missing samples)')

    # BMRT — LASER-KV defaults
    parser.add_argument('--method', default='hybrid', choices=['exact', 'lsh', 'hybrid'])
    parser.add_argument('--backend', default='flash', choices=['eager', 'flash'])
    parser.add_argument('--compression_mode', default='accumulate', choices=['accumulate', 'recursive'])
    parser.add_argument('--compression_ratio', type=float, default=0.75)
    parser.add_argument('--hybrid_primary', default='exact')
    parser.add_argument('--hybrid_secondary', default='lsh')
    parser.add_argument('--lsh_mode', default='magicpig_baseline',
                        choices=['frequency_rank', 'magicpig_baseline'])
    parser.add_argument('--hybrid_ratio', type=float, default=0.75)
    parser.add_argument('--protection_divisor', type=int, default=4)
    parser.add_argument('--block_size', type=int, default=4096)
    parser.add_argument('--num_bits', type=int, default=6)
    parser.add_argument('--num_tables', type=int, default=4)
    parser.add_argument('--stop_words', default='')
    parser.add_argument('--infer_script', default=None,
                        help='Override inference script (default: tests/pred_ruler.py)')
    parser.add_argument('--window_size', type=int, default=None,
                        help='PyramidKV window size; passed through to infer_script if set')

    args = parser.parse_args()
    main(args)
