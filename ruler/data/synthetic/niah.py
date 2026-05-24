# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Needle-in-a-haystack dataset generator."""
import argparse
import json
import os
import random
import re
import sys
import uuid
from pathlib import Path

import numpy as np
import wonderwords
from tqdm import tqdm
from utils import write_jsonl

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from nltk.tokenize import sent_tokenize
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
parser.add_argument('--num_needle_k', type=int, default=1)
parser.add_argument('--num_needle_v', type=int, default=1)
parser.add_argument('--num_needle_q', type=int, default=1)
parser.add_argument('--type_haystack', type=str, default='essay')
parser.add_argument('--type_needle_k', type=str, default='words')
parser.add_argument('--type_needle_v', type=str, default='numbers')

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)
args.num_needle_k = max(args.num_needle_k, args.num_needle_q)

TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)

needle = 'One of the special magic {type_needle_v} for {key} is: {value}.'
if args.type_haystack == 'essay':
    essay_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'json/PaulGrahamEssays.json')
    essay = json.load(open(essay_path))['text']
    haystack = re.sub(r'\s+', ' ', essay).split(' ')
elif args.type_haystack == 'repeat':
    haystack = 'The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.'
elif args.type_haystack == 'needle':
    haystack = needle
else:
    raise NotImplementedError(f'{args.type_haystack} is not implemented.')

nouns = wonderwords.random_word._get_words_from_text_file('nounlist.txt')
adjs = wonderwords.random_word._get_words_from_text_file('adjectivelist.txt')
words = sorted(list(set([f'{adj}-{noun}' for adj in adjs for noun in nouns])))

DEPTHS = list(np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int))


def generate_random_number(num_digits=7):
    return str(random.randint(10 ** (num_digits - 1), 10 ** num_digits - 1))


def generate_random_word():
    return random.choice(words)


def generate_random_uuid():
    return str(uuid.UUID(int=random.getrandbits(128), version=4))


def generate_random(type_needle):
    if type_needle == 'numbers':
        return generate_random_number()
    elif type_needle == 'words':
        return generate_random_word()
    elif type_needle == 'uuids':
        return generate_random_uuid()
    raise NotImplementedError(f'{type_needle} not implemented.')


def generate_input_output(num_haystack):
    keys, values, needles = [], [], []
    for _ in range(args.num_needle_k):
        keys.append(generate_random(args.type_needle_k))
        value = []
        for _ in range(args.num_needle_v):
            value.append(generate_random(args.type_needle_v))
            needles.append(needle.format(
                type_needle_v=args.type_needle_v,
                key=keys[-1],
                value=value[-1],
            ))
        values.append(value)

    random.Random(args.random_seed).shuffle(needles)

    if args.type_haystack == 'essay':
        text = ' '.join(haystack[:num_haystack])
        document_sents = sent_tokenize(text.strip())
        insertion_positions = (
            [0]
            + sorted([int(len(document_sents) * (d / 100)) for d in random.sample(DEPTHS, len(needles))])
            + [len(document_sents)]
        )
        parts = []
        for i in range(1, len(insertion_positions)):
            parts.append(' '.join(document_sents[insertion_positions[i-1]:insertion_positions[i]]))
            if i - 1 < len(needles):
                parts.append(needles[i - 1])
        context = ' '.join(parts)
    else:
        if args.type_haystack == 'repeat':
            sentences = [haystack] * num_haystack
        elif args.type_haystack == 'needle':
            sentences = [
                haystack.format(
                    type_needle_v=args.type_needle_v,
                    key=generate_random(args.type_needle_k),
                    value=generate_random(args.type_needle_v),
                )
                for _ in range(num_haystack)
            ]
        indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
        for index, element in zip(indexes, needles):
            sentences.insert(index, element)
        context = '\n'.join(sentences)

    indices = random.sample(range(args.num_needle_k), args.num_needle_q)
    queries = [keys[i] for i in indices]
    answers = [a for i in indices for a in values[i]]
    query = (', '.join(queries[:-1]) + ', and ' + queries[-1]) if len(queries) > 1 else queries[0]

    context_template, query_template = args.context_template, args.query_template
    type_needle_v = args.type_needle_v
    if args.num_needle_q * args.num_needle_v == 1:
        def _singular(t):
            return t.replace('Some', 'A').replace('are all', 'is').replace('are', 'is').replace('answers', 'answer')
        context_template = _singular(context_template)
        query_template = _singular(query_template)
        type_needle_v = type_needle_v[:-1]

    context = context_template.format(type_needle_v=type_needle_v, context=context)
    query = query_template.format(type_needle_v=type_needle_v, query=query)
    return context, query, answers


def generate_samples(num_samples, max_seq_length):
    write_jsons = []
    tokens_to_generate = args.tokens_to_generate

    incremental = 500 if args.type_haystack == 'essay' else 25
    if args.type_haystack != 'essay' and max_seq_length < 4096:
        incremental = 5

    num_haystack = incremental
    total_tokens = 0
    while total_tokens + tokens_to_generate < max_seq_length:
        context, query, answer = generate_input_output(num_haystack)
        total_tokens = len(TOKENIZER.text_to_tokens(context + query + ' '.join(answer)))
        print(f'Max {max_seq_length} | Current {total_tokens + tokens_to_generate} | Haystack: {num_haystack}')
        if total_tokens + tokens_to_generate > max_seq_length:
            num_haystack -= incremental
            break
        if args.type_haystack == 'essay' and num_haystack > len(haystack):
            num_haystack = len(haystack)
            break
        num_haystack += incremental

    print('Num haystack:', num_haystack)

    for index in tqdm(range(num_samples)):
        used_haystack = num_haystack
        while True:
            try:
                context, query, answer = generate_input_output(used_haystack)
                length = len(TOKENIZER.text_to_tokens(context + query)) + tokens_to_generate
                assert length <= max_seq_length
                break
            except Exception:
                if used_haystack > incremental:
                    used_haystack -= incremental

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
    write_jsonl(generate_samples(args.num_samples, args.max_seq_length), save_file)


if __name__ == '__main__':
    main()
