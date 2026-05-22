"""Speed + VRAM benchmark: dense flash-attn vs BMRT compression.

Times prefill+generate over (num_warmup + num_measure) BABILong samples.
Logs mean+std latency and peak VRAM per run to a results txt file.
"""
import argparse
import os
import sys
import time
import gc
import statistics

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bmrt import RecursiveCompressionEngine


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=['dense', 'bmrt'])
    p.add_argument('--model_path', default='meta-llama/Llama-3.1-8B-Instruct')
    p.add_argument('--dataset_config', required=True, help='16k/32k/64k/128k')
    p.add_argument('--dataset_split', default='qa1')
    p.add_argument('--num_warmup', type=int, default=5)
    p.add_argument('--num_measure', type=int, default=30)
    p.add_argument('--max_new_tokens', type=int, default=100)
    p.add_argument('--results_file', default='speed_results.txt')

    p.add_argument('--compression_ratio', type=float, default=0.25)
    p.add_argument('--block_size', type=int, default=4096)
    p.add_argument('--protection_divisor', type=int, default=4)
    p.add_argument('--hybrid_primary', default='exact')
    p.add_argument('--hybrid_secondary', default='lsh')
    p.add_argument('--hybrid_ratio', type=float, default=0.75)
    p.add_argument('--lsh_mode', default='magicpig_baseline')
    p.add_argument('--num_bits', type=int, default=10)
    p.add_argument('--num_tables', type=int, default=8)
    p.add_argument('--selector_mode', default='partitioned_centroid')
    return p.parse_args()


def build_dense(model_path):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation='flash_attention_2',
    )
    model.eval()
    return tok, model


@torch.no_grad()
def run_dense_sample(tok, model, context, query, max_new_tokens, device):
    ctx_ids = tok.encode(context, return_tensors='pt', add_special_tokens=False).to(device)
    q_ids = tok.apply_chat_template(
        [{"role": "user", "content": query}],
        add_generation_prompt=True, return_tensors='pt'
    ).to(device)
    input_ids = torch.cat([ctx_ids, q_ids], dim=1)

    out = model(input_ids=input_ids, use_cache=True)
    cache = out.past_key_values
    next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
    eos = tok.eos_token_id

    for _ in range(max_new_tokens - 1):
        out = model(input_ids=next_tok, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(-1)
        if next_tok.item() == eos:
            break

    del cache, out


def parse_split_size(dataset_config):
    return int(dataset_config.lower().rstrip('k')) * 1000


def compute_budget(ratio, block_size, split_size):
    return int((2 * ratio * block_size * split_size) / (block_size + split_size))


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading dataset RMT-team/babilong {args.dataset_config} {args.dataset_split}...")
    ds = load_dataset("RMT-team/babilong", args.dataset_config, split=args.dataset_split)
    total_needed = args.num_warmup + args.num_measure
    n = min(total_needed, len(ds))
    samples = [(ds[i]['input'], ds[i]['question']) for i in range(n)]
    if n < total_needed:
        print(f"WARNING: only {n} samples available, needed {total_needed}")

    if args.mode == 'dense':
        tok, model = build_dense(args.model_path)
        runner = lambda c, q: run_dense_sample(tok, model, c, q, args.max_new_tokens, device)
        config_str = "config=dense_flash_attn"
    else:
        split_size = parse_split_size(args.dataset_config)
        budget = compute_budget(args.compression_ratio, args.block_size, split_size)
        print(f"BMRT budget: {budget} (ratio={args.compression_ratio}, split={split_size})")
        engine = RecursiveCompressionEngine(
            model_path=args.model_path,
            selector_type='hybrid',
            lsh_mode=args.lsh_mode,
            selector_mode=args.selector_mode,
            compression_mode='accumulate',
            backend='flash',
            budget=budget,
            protection_divisor=args.protection_divisor,
            block_size=args.block_size,
            max_new_tokens=args.max_new_tokens,
            num_bits=args.num_bits,
            num_tables=args.num_tables,
            hybrid_primary=args.hybrid_primary,
            hybrid_secondary=args.hybrid_secondary,
            hybrid_ratio=args.hybrid_ratio,
        )
        runner = lambda c, q: engine(prompt_context=c, prompt_query=q)
        config_str = (
            f"compression_ratio={args.compression_ratio} | block_size={args.block_size} | "
            f"protection_divisor={args.protection_divisor} | "
            f"hybrid={args.hybrid_primary}+{args.hybrid_secondary}:{args.hybrid_ratio} | "
            f"lsh_mode={args.lsh_mode} | num_bits={args.num_bits} | num_tables={args.num_tables}"
        )

    print(f"Warmup: {args.num_warmup} samples...")
    for i in range(min(args.num_warmup, n)):
        c, q = samples[i]
        runner(c, q)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    print(f"Measuring: {args.num_measure} samples...")
    latencies = []
    for idx, i in enumerate(range(args.num_warmup, n)):
        c, q = samples[i]
        torch.cuda.synchronize()
        t0 = time.time()
        runner(c, q)
        torch.cuda.synchronize()
        dt = time.time() - t0
        latencies.append(dt)
        print(f"  [{idx+1}/{args.num_measure}] {dt:.2f}s")
        gc.collect()
        torch.cuda.empty_cache()

    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    mean_lat = statistics.mean(latencies) if latencies else 0.0
    std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    total_lat = sum(latencies)

    line = (
        f"mode={args.mode} | dataset={args.dataset_config}/{args.dataset_split} | "
        f"n_measure={len(latencies)} | n_warmup={args.num_warmup} | "
        f"mean_latency_s={mean_lat:.3f} | std_latency_s={std_lat:.3f} | "
        f"total_latency_s={total_lat:.2f} | peak_vram_gb={peak_gb:.2f} | "
        f"max_new_tokens={args.max_new_tokens} | model={args.model_path} | {config_str}\n"
    )

    with open(args.results_file, 'a', encoding='utf-8') as f:
        f.write(line)

    print("\n" + "=" * 60)
    print(line.strip())
    print(f"Logged to {args.results_file}")


if __name__ == '__main__':
    main()
