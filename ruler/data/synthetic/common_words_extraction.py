# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common words extraction dataset generator."""
import argparse
import os
import random
import sys
from pathlib import Path

import wonderwords
from tqdm import tqdm
from utils import write_jsonl

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from tokenizer import select_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=Path, required=True)
parser.add_argument('--save_name', type=str, required=True)
parser.add_argument('--subset', type=str, default='validation')
parser.add_argument('--tokenizer_path', type=str, required=True)
parser.add_argument('--tokenizer_type', type=str, default='hf')
parser.add_argument('--max_seq_length', type=int, required=True)
parser.add_argument('--tokens_to_generate', type=int, required=True)
parser.add_argument('--num_samples', type=int, required=True)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--context_template', type=str, default='')
parser.add_argument('--query_template', type=str, default='')
parser.add_argument('--remove_newline_tab', action='store_true')
parser.add_argument('--freq_cw', type=int, default=30)
parser.add_argument('--freq_ucw', type=int, default=3)
parser.add_argument('--num_cw', type=int, default=10)

args = parser.parse_args()
random.seed(args.random_seed)

TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)

nouns = wonderwords.random_word._get_words_from_text_file('nounlist.txt')
adjs = wonderwords.random_word._get_words_from_text_file('adjectivelist.txt')
verbs = wonderwords.random_word._get_words_from_text_file('verblist.txt')
words = sorted(list(set(nouns + adjs + verbs)))
random.Random(args.random_seed).shuffle(words)


def get_example(num_words, common_repeats=30, uncommon_repeats=3, common_nums=10):
    word_list_full = random.sample(words, num_words)
    common, uncommon = word_list_full[:common_nums], word_list_full[common_nums:]
    word_list = common * int(common_repeats) + uncommon * int(uncommon_repeats)
    random.Random(args.random_seed).shuffle(word_list)
    context = ' '.join([f'{i+1}. {word}' for i, word in enumerate(word_list)])
    return context, common


def generate_input_output(num_words):
    if args.max_seq_length < 4096:
        context_example, answer_example = get_example(20, 3, 1, args.num_cw)
        context, answer = get_example(num_words, 6, 1, args.num_cw)
    else:
        context_example, answer_example = get_example(40, 10, 3, args.num_cw)
        context, answer = get_example(num_words, args.freq_cw, args.freq_ucw, args.num_cw)

    context_template, input_query = args.context_template, args.query_template
    input_example = (
        context_template.format(context=context_example)
        + input_query
        + ' '.join([f'{i+1}. {word}' for i, word in enumerate(answer_example)])
    )
    input_context = context_template.format(context=context)
    return input_example + '\n' + input_context, input_query, answer


def sys_word_pair_random(num_samples, max_seq_length, incremental=10):
    write_jsons = []
    tokens_to_generate = args.tokens_to_generate
    num_words = incremental
    total_tokens = 0

    while total_tokens + tokens_to_generate < max_seq_length:
        context, query, answer = generate_input_output(num_words)
        total_tokens = len(TOKENIZER.text_to_tokens(
            context + query + ' ' + ' '.join([f'{i+1}. {w}' for i, w in enumerate(answer)])
        ))
        print(f'Max {max_seq_length} | Current {total_tokens + tokens_to_generate} | Words: {num_words}')
        if total_tokens + tokens_to_generate > max_seq_length:
            num_words -= incremental
            break
        num_words += incremental
        if num_words > len(words):
            num_words = len(words)
            break

    print('num_words:', num_words)

    for index in tqdm(range(num_samples)):
        used_words = num_words
        while True:
            try:
                context, query, answer = generate_input_output(used_words)
                length = len(TOKENIZER.text_to_tokens(context + query)) + tokens_to_generate
                assert length <= max_seq_length
                break
            except Exception:
                if used_words > incremental:
                    used_words -= incremental

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
    write_jsonl(sys_word_pair_random(args.num_samples, args.max_seq_length), save_file)


if __name__ == '__main__':
    main()
