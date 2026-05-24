# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frequent words extraction dataset generator."""
import argparse
import os
import random
import string
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
from utils import write_jsonl

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from scipy.special import zeta
from tokenizer import select_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=Path, required=True)
parser.add_argument('--save_name', type=str, required=True)
parser.add_argument('--subset', type=str, default='validation')
parser.add_argument('--tokenizer_path', type=str, required=True)
parser.add_argument('--tokenizer_type', type=str, default='hf')
parser.add_argument('--max_seq_length', type=int, required=True)
parser.add_argument('--tokens_to_generate', type=int, default=50)
parser.add_argument('--num_samples', type=int, required=True)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--context_template', type=str, default='')
parser.add_argument('--query_template', type=str, default='')
parser.add_argument('--remove_newline_tab', action='store_true')
parser.add_argument('--coded_wordlen', type=int, default=6)
parser.add_argument('--vocab_size', type=int, default=-1)
parser.add_argument('--alpha', type=float, default=2.0)
parser.add_argument('--add_fewshot', action='store_true', default=False)

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)

TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)


def generate_input_output(max_len, num_words=-1, coded_wordlen=6, vocab_size=2000, incremental=10, alpha=2.0):
    vocab = [''.join(random.choices(string.ascii_lowercase, k=coded_wordlen)) for _ in range(vocab_size)]
    while len(set(vocab)) < vocab_size:
        vocab.append(''.join(random.choices(string.ascii_lowercase, k=coded_wordlen)))
    vocab = sorted(list(set(vocab)))
    random.Random(args.random_seed).shuffle(vocab)
    vocab[0] = '...'

    context_template, input_query = args.context_template, args.query_template

    def gen_text(nw):
        k = np.arange(1, len(vocab) + 1)
        sampled_cnt = nw * (k ** -alpha) / zeta(alpha)
        sampled_words = [w for w, zi in zip(vocab, sampled_cnt.astype(int)) for _ in range(zi)]
        random.Random(args.random_seed).shuffle(sampled_words)
        return context_template.format(context=' '.join(sampled_words)), vocab[1:4]

    if num_words > 0:
        text, answer = gen_text(num_words)
        while len(TOKENIZER.text_to_tokens(text)) > max_len:
            num_words -= incremental
            text, answer = gen_text(num_words)
    else:
        num_words = max_len // coded_wordlen
        text, answer = gen_text(num_words)
        while len(TOKENIZER.text_to_tokens(text)) < max_len:
            num_words += incremental
            text, answer = gen_text(num_words)
        num_words -= incremental

    text, answer = gen_text(num_words)
    return text, input_query, answer, num_words


def sys_kwext(num_samples, max_seq_length, incremental=10):
    write_jsons = []
    tokens_to_generate = args.tokens_to_generate
    vocab_size = max_seq_length // 50 if args.vocab_size == -1 else args.vocab_size

    _, _, _, num_example_words = generate_input_output(
        max_seq_length, coded_wordlen=args.coded_wordlen,
        vocab_size=vocab_size, incremental=max_seq_length // 32, alpha=args.alpha,
    )
    print('num_example_words:', num_example_words)

    for index in tqdm(range(num_samples)):
        context, query, answer, _ = generate_input_output(
            max_seq_length, num_words=num_example_words,
            coded_wordlen=args.coded_wordlen, vocab_size=vocab_size,
            incremental=max_seq_length // 32, alpha=args.alpha,
        )
        length = len(TOKENIZER.text_to_tokens(context + query)) + tokens_to_generate

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
    write_jsonl(sys_kwext(args.num_samples, args.max_seq_length), save_file)


if __name__ == '__main__':
    main()
