# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate RULER predictions and write summary.csv + submission.csv.

Usage:
    python ruler/eval/evaluate.py \
        --data_dir /path/to/results/seq_len \
        --benchmark synthetic
"""
import argparse
import importlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import nltk
import pandas as pd
import yaml
from tqdm import tqdm

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVAL_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--benchmark', type=str, default='synthetic')
parser.add_argument('--verbose', type=int, default=0)
args = parser.parse_args()


def read_jsonl(filename, num_lines=-1):
    lines = []
    with open(filename) as f:
        for i, line in enumerate(f):
            lines.append(json.loads(line))
            if i == num_lines:
                break
    return lines


def write_jsonl(data, filename):
    with open(filename, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')


def postprocess_pred(predict_str, task_config):
    predict_str = predict_str.strip()
    predict_str = re.compile(r'[\x00-\x1f]').sub('\n', predict_str).strip()
    return predict_str


def get_pred_and_ref(predictions_file, task_config):
    lines = read_jsonl(predictions_file)
    inputs, predicts, references, indices = [], [], [], []
    for line in tqdm(lines):
        inputs.append(line['input_context'] + line['input_query'])
        predicts.append(postprocess_pred(line['pred'], task_config))
        references.append(line.get('outputs', [line.get('output', '')]))
        indices.append(line.get('others', {}).get('id', line['index']))
    return inputs, predicts, references, indices


def run_evaluation_per_task(task_config, predictions_file, verbose=0):
    inputs, predicts, references, indices = get_pred_and_ref(predictions_file, task_config)
    task_nulls = f'{sum(len(x) == 0 for x in predicts)}/{len(predicts)}'

    if references and references[0] and references[0][0] is not None:
        task_score = task_config['metric_fn'](predicts, references)
    else:
        task_score = 0.0

    if verbose:
        print('=' * 40)
        for i, (inp, ref, pred) in enumerate(zip(inputs, references, predicts)):
            print(f'Input     : {inp}')
            print(f'Reference : {ref}')
            print(f'Prediction: {pred}')
            print('=' * 40)
            if i > verbose:
                break

    return task_score, task_nulls, predicts, indices


def write_evaluation(results):
    tasks = list(results.keys())
    dfs = [
        ['Tasks'] + tasks,
        ['Score'] + [results[t]['score'] for t in tasks],
        ['Nulls'] + [results[t]['nulls'] for t in tasks],
    ]
    output_file = os.path.join(
        args.data_dir,
        'summary.csv' if len(tasks) > 1 else f'summary-{tasks[0]}.csv',
    )
    df = pd.DataFrame(dfs)
    df.to_csv(output_file, index=False)
    print('\n=============================================\n')
    print(df)
    print(f'\nSaved eval results to {output_file}')


def write_submission(results):
    COLUMNS = ['Task', 'ID', 'Prediction']
    dfs = pd.DataFrame(columns=COLUMNS)
    for task, result in results.items():
        df = pd.DataFrame({'Task': task, 'ID': result['indices'], 'Prediction': result['predicts']})
        dfs = pd.concat((dfs, df[COLUMNS]))
    output_file = os.path.join(args.data_dir, 'submission.csv')
    dfs.reset_index(drop=True).to_csv(output_file, index=False)
    print(f'Saved submission results to {output_file}')


def aggregate_chunk(folder):
    jsonl_files = [f for f in os.listdir(folder) if Path(f).suffix == '.jsonl']
    chunk_files = sorted([f for f in jsonl_files if re.match(r'.*[^_]+-\d+\.jsonl', f)])
    chunk_dict = defaultdict(list)
    for f in chunk_files:
        task = '-'.join(f.split('-')[:-1])
        chunk_dict[task].append(f)
    for task, files in chunk_dict.items():
        lines = []
        for f in sorted(files):
            fpath = os.path.join(folder, f)
            lines += read_jsonl(fpath)
            os.remove(fpath)
        write_jsonl(lines, os.path.join(folder, f'{task}.jsonl'))


def main():
    ruler_dir = os.path.dirname(EVAL_DIR)
    module = importlib.import_module(f'{args.benchmark}.constants')
    tasks_base = module.TASKS

    with open(os.path.join(ruler_dir, f'{args.benchmark}_task_config.yaml')) as f:
        tasks_customized = yaml.safe_load(f)

    TASKS = tasks_customized
    for _, config in TASKS.items():
        config.update(tasks_base[config['task']])

    print(f'Total tasks: {list(TASKS.keys())}')
    aggregate_chunk(args.data_dir)

    jsonl_files = [f for f in os.listdir(args.data_dir) if Path(f).suffix == '.jsonl']
    eval_results, subm_results = {}, {}

    for task, config in TASKS.items():
        if f'{task}.jsonl' not in jsonl_files:
            print(f'Prediction file {task}.jsonl not found — skipping.')
            continue
        print(f'Evaluate task {task}...')
        task_score, task_nulls, predicts, indices = run_evaluation_per_task(
            predictions_file=os.path.join(args.data_dir, f'{task}.jsonl'),
            task_config=config,
            verbose=args.verbose,
        )
        eval_results[task] = {'score': task_score, 'nulls': task_nulls}
        subm_results[task] = {'predicts': predicts, 'indices': indices}

    write_evaluation(eval_results)
    write_submission(subm_results)


if __name__ == '__main__':
    main()
