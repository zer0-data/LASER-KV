"""SnapKV inference on RULER benchmark data."""
import argparse
import gc
import json
import os
import sys

import torch
from transformers import pipeline
from kvpress import SnapKVPress


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_done_indices(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)['index'] for line in f}


def main(args):
    print('=' * 60)
    print('  SnapKV RULER Inference')
    print('=' * 60)
    print(f'  Model:             {args.model_path}')
    print(f'  Compression ratio: {args.compression_ratio}')
    print(f'  Input:             {args.input_path}')
    print(f'  Output:            {args.output_path}')
    print('=' * 60)

    samples = read_jsonl(args.input_path)
    if args.num_samples > 0:
        samples = samples[:args.num_samples]

    done_indices = load_done_indices(args.output_path) if args.use_cache else set()
    if done_indices:
        print(f'Resuming: {len(done_indices)} samples already done.')

    press = SnapKVPress(compression_ratio=args.compression_ratio)
    pipe = pipeline(
        'kv-press-text-generation',
        model=args.model_path,
        device='cuda:0',
        model_kwargs={'attn_implementation': 'flash_attention_2'},
        torch_dtype='auto',
        trust_remote_code=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    new_preds = 0
    for idx, sample in enumerate(samples):
        sys.stdout.write(f'\rProgress: [{idx+1}/{len(samples)}] ({(idx+1)/len(samples)*100:.1f}%)')
        sys.stdout.flush()

        sample_idx = sample['index']
        if sample_idx in done_indices:
            continue

        prompt = sample['input_context'] + sample['input_query']
        pred = ''
        try:
            output = pipe(
                prompt,
                press=press,
                max_new_tokens=args.tokens_to_generate,
                do_sample=False,
                temperature=1.0,
                return_full_text=False,
            )
            if isinstance(output, list) and 'generated_text' in output[0]:
                pred = output[0]['generated_text']
            else:
                pred = str(output)
        except Exception as e:
            print(f'\nError on sample {sample_idx}: {e}')
            import traceback
            traceback.print_exc()
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        record = {
            'index': sample_idx,
            'pred': pred,
            'input_context': sample['input_context'],
            'input_query': sample['input_query'],
            'outputs': sample['outputs'],
            'others': sample.get('others', {'id': sample_idx}),
        }
        with open(args.output_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        new_preds += 1

    print(f'\nDone. {new_preds} new predictions written to {args.output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # I/O
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--use_cache', action='store_true')
    # Model
    parser.add_argument('--model_path', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--tokens_to_generate', type=int, default=128)
    parser.add_argument('--seq_length', type=int, default=4096)
    # SnapKV
    parser.add_argument('--compression_ratio', type=float, default=0.25)
    # Accepted but unused — keeps run_ruler.py passthrough compatible
    parser.add_argument('--method', default='snapkv')
    parser.add_argument('--backend', default='flash')
    parser.add_argument('--compression_mode', default='accumulate')
    parser.add_argument('--lsh_mode', default='magicpig_baseline')
    parser.add_argument('--selector_mode', default='l2')
    parser.add_argument('--hybrid_primary', default='exact')
    parser.add_argument('--hybrid_secondary', default='lsh')
    parser.add_argument('--hybrid_ratio', type=float, default=0.75)
    parser.add_argument('--protection_divisor', type=int, default=4)
    parser.add_argument('--block_size', type=int, default=4096)
    parser.add_argument('--num_bits', type=int, default=6)
    parser.add_argument('--num_tables', type=int, default=4)
    parser.add_argument('--stop_words', default='')
    parser.add_argument('--budget', type=int, default=0)
    args = parser.parse_args()
    main(args)
