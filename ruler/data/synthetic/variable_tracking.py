# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Variable tracking dataset generator."""
import argparse
import os
import random
import string
import sys
from pathlib import Path

from constants import TASKS
from tqdm import tqdm
from utils import write_jsonl

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from tokenizer import select_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=Path, required=True)
parser.add_argument('--save_name', type=str, required=True)
parser.add_argument('--subset', type=str, default='validation')
parser.add_argument('--tokenizer_path', type=str, required=True)
parser.add_argument('--tokenizer_type', type=str, default='hf')
parser.add_argument('--max_seq_length', type=int, required=True)
parser.add_argument('--tokens_to_generate', type=int, default=120)
parser.add_argument('--num_samples', type=int, required=True)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--context_template', type=str, default='')
parser.add_argument('--query_template', type=str, default='')
parser.add_argument('--remove_newline_tab', action='store_true')
parser.add_argument('--num_chains', type=int, default=1)
parser.add_argument('--num_hops', type=int, default=4)
parser.add_argument('--add_fewshot', action='store_true', default=False)

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)

TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)


def generate_chains(num_chains, num_hops, is_icl=False):
    k = 5 if not is_icl else 3
    num_hops = num_hops if not is_icl else min(10, num_hops)
    vars_all = [''.join(random.choices(string.ascii_uppercase, k=k)).upper()
                for _ in range((num_hops + 1) * num_chains)]
    while len(set(vars_all)) < num_chains * (num_hops + 1):
        vars_all.append(''.join(random.choices(string.ascii_uppercase, k=k)).upper())

    vars_ret, chains_ret = [], []
    for i in range(0, len(vars_all), num_hops + 1):
        this_vars = vars_all[i:i + num_hops + 1]
        vars_ret.append(this_vars)
        this_chain = [f'VAR {this_vars[0]} = {np.random.randint(10000, 99999)}']
        for j in range(num_hops):
            this_chain.append(f'VAR {this_vars[j+1]} = VAR {this_vars[j]} ')
        chains_ret.append(this_chain)
    return vars_ret, chains_ret


def generate_input_output(num_noises, num_chains, num_hops, is_icl=False):
    vars, chains = generate_chains(num_chains, num_hops, is_icl=is_icl)
    noise = 'The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.\n'
    sentences = [noise] * num_noises

    if len(sentences) <= len(chains[0]):
        sentences = [n + '.' if n.strip() else n for n in [x for noise in sentences for x in noise.split('.')]]
        try:
            assert len(sentences) > len(chains[0])
        except Exception:
            print('reduces chain length for not enough noises')
            chains = [chain[:len(sentences) - 1] for chain in chains]

    for chain_i in chains:
        positions = list(sorted(random.sample(range(len(sentences)), len(chain_i))))
        for insert_pi, j in zip(positions, range(len(chain_i))):
            sentences.insert(insert_pi + j, chain_i[j])

    context = ' '.join(sentences).replace('. \n', '.\n')

    context_template, query_template = args.context_template, args.query_template
    if is_icl and context_template != TASKS['variable_tracking']['context_template']:
        cutoff = context_template.index(TASKS['variable_tracking']['context_template'][:20])
        context_template = ' '.join(context_template[cutoff:].split()) + ' '
        cutoff_ans = query_template.index(TASKS['variable_tracking']['answer_prefix'][:10])
        query_template = ' '.join(query_template[:cutoff_ans].split()[:-1]) + query_template[cutoff_ans:]

    value = chains[0][0].split('=')[-1].strip()
    context = context_template.format(context=context)
    query = query_template.format(query=value, num_v=num_hops + 1)
    return context, query, vars[0]


def randomize_icl(icl_example):
    icl_tgt_cut = icl_example.index(TASKS['variable_tracking']['answer_prefix'][-10:])
    icl_tgt = icl_example[icl_tgt_cut + 10:].strip().split()
    for item in icl_tgt:
        new_item = ''.join(random.choices(string.ascii_uppercase, k=len(item))).upper()
        icl_example = icl_example.replace(item, new_item)
    return icl_example


def sys_vartrack_w_noise_random(num_samples, max_seq_length, incremental=10, num_chains=1,
                                 num_hops=4, add_fewshot=True, icl_example=None):
    write_jsons = []
    tokens_to_generate = args.tokens_to_generate
    num_noises = incremental
    total_tokens = 0
    example_tokens = 0

    if add_fewshot and icl_example is not None:
        icl_out = ' '.join(icl_example['outputs'])
        icl_str = icl_example['input_context'] + icl_example['input_query'] + ' ' + icl_out + '\n\n'
        example_tokens = len(TOKENIZER.text_to_tokens(icl_str))

    while total_tokens + tokens_to_generate + example_tokens < max_seq_length:
        context, query, answer = generate_input_output(
            num_noises, num_chains, num_hops, is_icl=add_fewshot & (icl_example is None)
        )
        total_tokens = len(TOKENIZER.text_to_tokens(context + query + f' {answer}'))
        print(f'Max {max_seq_length} | Current {total_tokens + tokens_to_generate + example_tokens} | Noises: {num_noises}')
        if total_tokens + tokens_to_generate + example_tokens > max_seq_length:
            num_noises -= incremental
            break
        num_noises += incremental

    print('Num noises:', num_noises)

    for index in tqdm(range(num_samples)):
        used_noises = num_noises
        while True:
            try:
                context, query, answer = generate_input_output(
                    num_noises, num_chains, num_hops, is_icl=add_fewshot & (icl_example is None)
                )
                length = len(TOKENIZER.text_to_tokens(context + query)) + tokens_to_generate + example_tokens
                assert length <= max_seq_length
                break
            except Exception:
                if used_noises > incremental:
                    used_noises -= incremental

        if add_fewshot and icl_example is not None:
            cutoff = context.index(TASKS['variable_tracking']['context_template'][:20])
            icl_str = icl_example['input_context'] + icl_example['input_query'] + ' ' + ' '.join(icl_example['outputs']) + '\n\n'
            context = context[:cutoff] + ' ' + randomize_icl(icl_str) + '\n\n' + context[cutoff:]

        if args.remove_newline_tab:
            context = ' '.join(context.replace('\n', ' ').replace('\t', ' ').strip().split())
            query = ' '.join(query.replace('\n', ' ').replace('\t', ' ').strip().split())

        write_jsons.append({
            'index': index,
            'input_context': context,
            'input_query': query,
            'outputs': answer,
            'length': length,
        })

    return write_jsons


def main():
    save_file = args.save_dir / args.save_name / f'{args.subset}.jsonl'
    save_file.parent.mkdir(parents=True, exist_ok=True)

    icl_example = sys_vartrack_w_noise_random(
        num_samples=1, max_seq_length=500, incremental=5,
        num_chains=args.num_chains, num_hops=args.num_hops,
    )[0]
    write_jsons = sys_vartrack_w_noise_random(
        num_samples=args.num_samples, max_seq_length=args.max_seq_length,
        num_chains=args.num_chains, num_hops=args.num_hops, icl_example=icl_example,
    )
    write_jsonl(write_jsons, save_file)


if __name__ == '__main__':
    main()
