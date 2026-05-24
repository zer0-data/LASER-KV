"""BMRT inference on RULER benchmark data."""
import argparse
import gc
import json
import os
import sys
import time

import torch

from bmrt import RecursiveCompressionEngine


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
    print('  BMRT RULER Inference')
    print('=' * 60)
    print(f'  Model:         {args.model_path}')
    print(f'  Method:        {args.method} (backend={args.backend})')
    print(f'  Compression:   {args.compression_mode}, ratio={args.compression_ratio}')
    if args.method in ('lsh', 'hybrid'):
        print(f'  LSH:           mode={args.lsh_mode}, bits={args.num_bits}, tables={args.num_tables}')
    if args.method == 'hybrid':
        print(f'  Hybrid:        {args.hybrid_primary}+{args.hybrid_secondary} @ ratio={args.hybrid_ratio}')
    print(f'  Input:         {args.input_path}')
    print(f'  Output:        {args.output_path}')
    print('=' * 60)

    samples = read_jsonl(args.input_path)
    if args.num_samples > 0:
        samples = samples[:args.num_samples]

    done_indices = load_done_indices(args.output_path) if args.use_cache else set()
    if done_indices:
        print(f'Resuming: {len(done_indices)} samples already done.')

    # Budget: fixed override > per-sample length > seq_length arg
    if args.budget > 0:
        budget = args.budget
        print(f'Budget: {budget} (fixed)')
    else:
        sample_len = samples[0].get('length', args.seq_length)
        if args.compression_mode == 'recursive':
            budget = int(args.compression_ratio * sample_len)
        else:
            budget = int(
                (2 * args.compression_ratio * args.block_size * sample_len)
                / (args.block_size + sample_len)
            )
        print(f'Budget: {budget} (computed from seq_len={sample_len}, ratio={args.compression_ratio})')

    def build_engine():
        return RecursiveCompressionEngine(
            model_path=args.model_path,
            selector_type=args.method,
            lsh_mode=args.lsh_mode,
            selector_mode=args.selector_mode,
            compression_mode=args.compression_mode,
            backend=args.backend,
            budget=budget,
            protection_divisor=args.protection_divisor,
            block_size=args.block_size,
            max_new_tokens=args.tokens_to_generate,
            stop_words=args.stop_words.split(',') if args.stop_words else None,
            num_bits=args.num_bits,
            num_tables=args.num_tables,
            hybrid_primary=args.hybrid_primary,
            hybrid_secondary=args.hybrid_secondary,
            hybrid_ratio=args.hybrid_ratio,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    new_preds = 0
    for idx, sample in enumerate(samples):
        sys.stdout.write(f'\rProgress: [{idx+1}/{len(samples)}] ({(idx+1)/len(samples)*100:.1f}%)')
        sys.stdout.flush()

        sample_idx = sample['index']
        if sample_idx in done_indices:
            continue

        # Reload engine per sample for state isolation (slower but stable)
        engine = build_engine()
        pred = ''
        try:
            result = engine(
                prompt_context=sample['input_context'],
                prompt_query=sample['input_query'],
            )
            pred = result['text'][0]
        except Exception as e:
            print(f'\nError on sample {sample_idx}: {e}')
            import traceback
            traceback.print_exc()
        finally:
            try:
                engine.cleanup()
            except Exception:
                pass
            del engine
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
    parser.add_argument('--input_path', required=True, help='RULER validation.jsonl')
    parser.add_argument('--output_path', required=True, help='Output predictions.jsonl')
    parser.add_argument('--num_samples', type=int, default=-1, help='Limit samples (-1 = all)')
    parser.add_argument('--use_cache', action='store_true', help='Resume from existing output')

    # Model
    parser.add_argument('--model_path', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--tokens_to_generate', type=int, default=128)
    parser.add_argument('--seq_length', type=int, default=4096,
                        help='Context length used for budget computation when --budget not set')

    # BMRT (LASER-KV defaults)
    parser.add_argument('--method', default='hybrid', choices=['exact', 'lsh', 'hybrid'])
    parser.add_argument('--lsh_mode', default='magicpig_baseline',
                        choices=['frequency_rank', 'magicpig_baseline'])
    parser.add_argument('--selector_mode', default='l2',
                        choices=['l2', 'max_sim', 'mahalanobis', 'partitioned_centroid', 'none'])
    parser.add_argument('--compression_mode', default='accumulate', choices=['accumulate', 'recursive'])
    parser.add_argument('--backend', default='flash', choices=['eager', 'flash'])
    parser.add_argument('--compression_ratio', type=float, default=0.75)
    parser.add_argument('--budget', type=int, default=0,
                        help='Fixed token budget; overrides compression_ratio if > 0')
    parser.add_argument('--protection_divisor', type=int, default=4)
    parser.add_argument('--block_size', type=int, default=4096)
    parser.add_argument('--stop_words', default='')

    # LSH
    parser.add_argument('--num_bits', type=int, default=6)
    parser.add_argument('--num_tables', type=int, default=4)

    # Hybrid
    parser.add_argument('--hybrid_primary', default='exact')
    parser.add_argument('--hybrid_secondary', default='lsh')
    parser.add_argument('--hybrid_ratio', type=float, default=0.75)

    args = parser.parse_args()
    main(args)
