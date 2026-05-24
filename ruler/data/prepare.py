# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Prepare jsonl with fields `input_context`, `input_query`, and `outputs`.

Usage:
    python ruler/data/prepare.py \
        --save_dir <path> \
        --task <task_name> \
        --tokenizer_path <model_path> \
        --tokenizer_type hf \
        --max_seq_length <seq_len> \
        --model_template_type llama3 \
        --num_samples 500
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from synthetic.constants import TASKS
from template import PROMPT_TEMPLATES

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=Path, required=True)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--tokenizer_path', type=str, required=True)
parser.add_argument('--tokenizer_type', type=str, default='hf')
parser.add_argument('--max_seq_length', type=int, required=True)
parser.add_argument('--num_samples', type=int, default=500)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--model_template_type', type=str, default='base')
parser.add_argument('--remove_newline_tab', action='store_true')
parser.add_argument('--chunk_idx', type=int, default=0)
parser.add_argument('--chunk_amount', type=int, default=1)

args = parser.parse_args()


def main():
    start_time = time.time()

    with open(os.path.join(os.path.dirname(CURRENT_DIR), 'synthetic_task_config.yaml')) as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in synthetic_task_config.yaml')

    config = tasks_customized.get(args.task)
    config.update(TASKS[config['task']])

    assert args.model_template_type in PROMPT_TEMPLATES, (
        f'{args.model_template_type} not in {list(PROMPT_TEMPLATES.keys())}'
    )
    model_template = PROMPT_TEMPLATES[args.model_template_type]['template']

    answer_prefix = config.get('answer_prefix', '')
    model_template_context, model_template_query = model_template.split('{task_template}')
    config['context_template'] = model_template_context + config['context_template']
    config['query_template'] = config['query_template'] + model_template_query + answer_prefix

    chunks = [
        (args.num_samples // args.chunk_amount) + (1 if i < args.num_samples % args.chunk_amount else 0)
        for i in range(args.chunk_amount)
    ]
    num_samples = chunks[args.chunk_idx]
    pre_samples = sum(chunks[:args.chunk_idx])
    random_seed = 42 + args.chunk_idx

    try:
        script = os.path.join(CURRENT_DIR, 'synthetic', f"{config['task']}.py")
        additional_args = ' '.join([f'--{k} {v}' for k, v in config['args'].items()])
        command = (
            f'python3 {script}'
            f' --save_dir {args.save_dir}'
            f' --save_name {args.task}'
            f' --subset validation'
            f' --tokenizer_path {args.tokenizer_path}'
            f' --tokenizer_type {args.tokenizer_type}'
            f' --max_seq_length {args.max_seq_length}'
            f' --tokens_to_generate {config["tokens_to_generate"]}'
            f' --num_samples {num_samples}'
            f' --random_seed {random_seed}'
            f' {additional_args}'
            + (f' --remove_newline_tab' if args.remove_newline_tab else '')
            + (f' --pre_samples {pre_samples}' if config['task'] == 'qa' else '')
            + f' --context_template "{config["context_template"]}"'
            + f' --query_template "{config["query_template"]}"'
        )
        print(command)
        result = subprocess.run(
            command, shell=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            executable='/bin/bash',
        )
        print(result.stdout)
        if result.returncode != 0:
            print("Error:", result.stderr)
            raise RuntimeError(f"Data generation failed for {args.task}")
    except subprocess.CalledProcessError as e:
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        raise

    save_file = args.save_dir / args.task / 'validation.jsonl'
    print(f"Prepared {args.task} -> {save_file}")
    print(f"Time: {round((time.time() - start_time) / 60, 1)} min")


if __name__ == '__main__':
    main()
