# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QA dataset generator (SQuAD / HotpotQA)."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
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
parser.add_argument('--pre_samples', type=int, default=0)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--context_template', type=str, default='')
parser.add_argument('--query_template', type=str, default='')
parser.add_argument('--remove_newline_tab', action='store_true')
parser.add_argument('--dataset', type=str, required=True)

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)

TOKENIZER = select_tokenizer(args.tokenizer_type, args.tokenizer_path)


def read_squad(file):
    with open(file) as f:
        data = json.load(f)
    total_docs = [p['context'] for d in data['data'] for p in d['paragraphs']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}
    total_qas = []
    for d in data['data']:
        more_docs = [total_docs_dict[p['context']] for p in d['paragraphs']]
        for p in d['paragraphs']:
            for qas in p['qas']:
                if not qas['is_impossible']:
                    total_qas.append({
                        'query': qas['question'],
                        'outputs': [a['text'] for a in qas['answers']],
                        'context': [total_docs_dict[p['context']]],
                        'more_context': [idx for idx in more_docs if idx != total_docs_dict[p['context']]],
                    })
    return total_qas, total_docs


def read_hotpotqa(file):
    with open(file) as f:
        data = json.load(f)
    total_docs = [f"{t}\n{''.join(p)}" for d in data for t, p in d['context']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}
    total_qas = []
    for d in data:
        total_qas.append({
            'query': d['question'],
            'outputs': [d['answer']],
            'context': [total_docs_dict[f"{t}\n{''.join(p)}"] for t, p in d['context']],
        })
    return total_qas, total_docs


DOCUMENT_PROMPT = 'Document {i}:\n{document}'
json_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'json')
if args.dataset == 'squad':
    QAS, DOCS = read_squad(os.path.join(json_dir, 'squad.json'))
    DOCS = DOCS * 5
elif args.dataset == 'hotpotqa':
    QAS, DOCS = read_hotpotqa(os.path.join(json_dir, 'hotpotqa.json'))
else:
    raise NotImplementedError(f'{args.dataset} not implemented.')


def generate_input_output(index, num_docs):
    curr_q = QAS[index]['query']
    curr_a = QAS[index]['outputs']
    curr_docs = QAS[index]['context']
    curr_more = QAS[index].get('more_context', [])

    if num_docs < len(DOCS):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [i for i, _ in enumerate(DOCS) if i not in curr_docs + curr_more]
            all_docs = curr_docs + curr_more + random.sample(
                addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more))
            )
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))
        all_docs = [DOCS[idx] for idx in all_docs]
    else:
        all_docs = DOCS

    random.Random(args.random_seed).shuffle(all_docs)
    context = '\n\n'.join([DOCUMENT_PROMPT.format(i=i+1, document=d) for i, d in enumerate(all_docs)])
    input_context = args.context_template.format(context=context)
    input_query = args.query_template.format(query=curr_q)
    return input_context, input_query, curr_a


def generate_samples(num_samples, max_seq_length, incremental=10):
    write_jsons = []
    tokens_to_generate = args.tokens_to_generate
    num_docs = incremental

    total_tokens = 0
    while total_tokens + tokens_to_generate < max_seq_length:
        context, query, answer = generate_input_output(0, num_docs)
        total_tokens = len(TOKENIZER.text_to_tokens(context + query + f' {answer}'))
        print(f'Max {max_seq_length} | Current {total_tokens + tokens_to_generate} | Docs: {num_docs}')
        if total_tokens + tokens_to_generate > max_seq_length:
            num_docs -= incremental
            break
        num_docs += incremental
        if num_docs > len(DOCS):
            num_docs = len(DOCS)
            break

    print('Number of documents:', num_docs)

    for index in tqdm(range(num_samples)):
        used_docs = num_docs
        while True:
            try:
                context, query, answer = generate_input_output(index + args.pre_samples, used_docs)
                length = len(TOKENIZER.text_to_tokens(context + query)) + tokens_to_generate
                assert length <= max_seq_length
                break
            except Exception:
                if used_docs > incremental:
                    used_docs -= incremental

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
