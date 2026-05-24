# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate RULER results across sequence lengths and print a table."""
import argparse
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_summary_file(seq_result_dir):
    with open(os.path.join(seq_result_dir, 'summary.csv')) as f:
        lines = f.readlines()
    tasks = lines[1].strip().split(',')[1:]
    scores = lines[2].strip().split(',')[1:]
    nulls = lines[3].strip().split(',')[1:]
    # Detect unfinished tasks: parse "X/Y" denominator and check vs max across tasks
    denoms = []
    for n in nulls:
        try:
            denoms.append(int(n.split('/')[1]))
        except (IndexError, ValueError):
            denoms.append(0)
    expected = max(denoms) if denoms else 0
    unfinished = [tasks[i] for i in range(len(nulls)) if denoms[i] != expected]
    return tasks, scores, nulls, unfinished


def display_results(seq_len_results, seq_len):
    r = seq_len_results[seq_len]
    print(f'\nSequence Length: {seq_len}')
    w = max(len(t) for t in r['tasks'])
    print(' | '.join(f'{t:^{w}}' for t in r['tasks']))
    print('-' * (len(r['tasks']) * (w + 3) - 3))
    print(' | '.join(f'{s:^{w}}' for s in r['scores']))
    print(' | '.join(f'{n:^{w}}' for n in r['nulls']))
    if r['unfinished']:
        print(f'\n== Unfinished: {", ".join(r["unfinished"])} ==\n')


def gather_experiment_results(exp_dir):
    missing = []
    results = {}
    for seq_len in sorted([x for x in os.listdir(exp_dir) if x.isdigit()], key=int):
        if not os.path.exists(os.path.join(exp_dir, seq_len, 'summary.csv')):
            missing.append(seq_len)
            continue
        tasks, scores, nulls, unfinished = parse_summary_file(os.path.join(exp_dir, seq_len))
        results[seq_len] = {'tasks': tasks, 'scores': scores, 'nulls': nulls, 'unfinished': unfinished}
        display_results(results, seq_len)

    if missing:
        print(f'\n++ Missing sequence lengths: {", ".join(missing)} ++')

    print('\nFull CSV:')
    for k, v in results.items():
        print(f'{",".join([k] + v["scores"])}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', required=True,
                        help='experiment name or path to results directory')
    parser.add_argument('--output_dir', default=os.path.join(BASE_DIR, 'results'))
    args = parser.parse_args()

    if '/' not in args.exp and '\\' not in args.exp:
        args.exp = os.path.join(args.output_dir, args.exp)
    if not os.path.isdir(args.exp):
        raise NotADirectoryError(f'Invalid experiment path: {args.exp}')

    gather_experiment_results(args.exp)
