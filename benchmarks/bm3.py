#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
改进的 Flash Attention Benchmark 示例
增加了控制内存清空、支持模拟更真实的非连续内存布局以及按 chunk 分块生成 KV 数据的功能。

使用示例：
  python bm3.py --iters 100 --batch-size 1 --seqlen-q 1024 --seqlen-k 1024 --nheads 12 --headdim 64 --spread 1 --continuous --use-chunk --num-chunks 4 --init-text-length 1024 --chunk-kv 256 --clear-cache True
"""
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 假设 flash_attn 模块已经加入到 PYTHONPATH 中
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func, flash_attn_with_kvcache

def generate_tensor(batch_size, seqlen, nheads, headdim, dtype=torch.float16, device='cuda', continuous=True, spread=1):
    """
    根据 continuous 与 spread 参数生成基本张量。
    如果 continuous 为 True 或 spread<=1，则生成连续内存张量；
    否则生成一个稍长的张量，并按间隔切片后返回，模拟非连续内存分布。
    """
    if continuous or spread <= 1:
        return torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    else:
        full_tensor = torch.randn(batch_size, seqlen * spread, nheads, headdim, dtype=dtype, device=device)
        return full_tensor[:, ::spread, :, :]

def generate_qkv_tensor(batch_size, seqlen, nheads, headdim, dtype=torch.float16, device='cuda', continuous=True, spread=1):
    """
    生成形状为 (B, seqlen, 3, nheads, headdim) 的 QKV 张量，并支持连续和非连续内存两种模式。
    """
    if continuous or spread <= 1:
        return torch.randn(batch_size, seqlen, 3, nheads, headdim, dtype=dtype, device=device)
    else:
        full_tensor = torch.randn(batch_size, seqlen * spread, 3, nheads, headdim, dtype=dtype, device=device)
        return full_tensor[:, ::spread, :, :, :]

def generate_tensor_by_chunks(batch_size, total_len, nheads, headdim, num_chunks, dtype=torch.float16, device='cuda'):
    """
    模拟实际 kvcache 的非连续内存布局，将总长度 total_len 按 num_chunks 划分，
    每个 chunk 长度 = total_len / num_chunks（这里假设能整除）。
    最后将每个 chunk 分别生成随机数据，再拼接成一个非连续的张量。
    """
    assert total_len % num_chunks == 0, "total_len 必须能被 num_chunks 整除"
    chunk_len = total_len // num_chunks
    chunk_list = []
    for _ in range(num_chunks):
        # 每个 chunk 随机生成后，不做连续性合并，依次存入 list
        chunk = torch.randn(batch_size, chunk_len, nheads, headdim, dtype=dtype, device=device)
        chunk_list.append(chunk)
    # 用 cat 拼接后的 tensor在内存上不一定是连续的（特别是当各 chunk 内存并非紧邻分布时）
    return torch.cat(chunk_list, dim=1)

def run_one_benchmark_config(config, continuous, spread, num_iters=100, clear_cache=True):
    """
    针对给定配置运行三种 Flash Attention 实现的 benchmark，并计算平均核时间（ms）。
    新增了 clear_cache 参数，可以控制是否每次迭代后清空内存。
    """
    batch_size = config['batch_size']
    seqlen_q   = config['seqlen_q']
    seqlen_k   = config['seqlen_k']
    nheads     = config['nheads']
    headdim    = config['headdim']
    dtype      = config.get('dtype', torch.float16)
    device     = config.get('device', 'cuda')
    
    layout_str = "Contiguous" if continuous else f"Non-Contiguous (spread={spread})"
    print(f">> Testing {layout_str} layout, configuration: {config.get('name','Unknown')}")
    print("--------------------------------------------")
    
    # 用 CUDA 事件进行计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)
    
    results = {}
    warmup_times = {}
    
    # -----------------------
    # Benchmark 1: Basic Attention
    # -----------------------
    # 对于非连续内存，仍可使用 generate_tensor;
    q = generate_tensor(batch_size, seqlen_q, nheads, headdim, dtype, device, continuous, spread)
    k = generate_tensor(batch_size, seqlen_k, nheads, headdim, dtype, device, continuous, spread)
    v = generate_tensor(batch_size, seqlen_k, nheads, headdim, dtype, device, continuous, spread)
    torch.cuda.synchronize()
    start_event.record()
    _ = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
    end_event.record()
    torch.cuda.synchronize()
    warmup_time_basic = start_event.elapsed_time(end_event)
    warmup_times['Basic Attention'] = warmup_time_basic
    if clear_cache:
        torch.cuda.empty_cache()
    
    total_time_basic = 0.0
    successful_iters = 0
    for _ in tqdm(range(num_iters), desc="Basic Attention iterations", leave=False):
        try:
            q = generate_tensor(batch_size, seqlen_q, nheads, headdim, dtype, device, continuous, spread)
            k = generate_tensor(batch_size, seqlen_k, nheads, headdim, dtype, device, continuous, spread)
            v = generate_tensor(batch_size, seqlen_k, nheads, headdim, dtype, device, continuous, spread)
            torch.cuda.synchronize()
            start_event.record()
            _ = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
            end_event.record()
            torch.cuda.synchronize()
            total_time_basic += start_event.elapsed_time(end_event)
            successful_iters += 1
        except torch.cuda.OutOfMemoryError as e:
            print("Basic Attention OOM error:", e)
            if clear_cache:
                torch.cuda.empty_cache()
            continue
        finally:
            if clear_cache:
                torch.cuda.empty_cache()
    avg_basic = total_time_basic / successful_iters if successful_iters > 0 else float('nan')
    results['Basic Attention'] = avg_basic

    # -----------------------
    # Benchmark 2: Packed QKV Attention
    # -----------------------
    qkv = generate_qkv_tensor(batch_size, seqlen_k, nheads, headdim, dtype, device, continuous, spread)
    torch.cuda.synchronize()
    start_event.record()
    _ = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False)
    end_event.record()
    torch.cuda.synchronize()
    warmup_time_packed = start_event.elapsed_time(end_event)
    warmup_times['Packed QKV Attention'] = warmup_time_packed
    if clear_cache:
        torch.cuda.empty_cache()

    total_time_packed = 0.0
    successful_iters = 0
    for _ in tqdm(range(num_iters), desc="Packed QKV Attention iterations", leave=False):
        try:
            qkv = generate_qkv_tensor(batch_size, seqlen_k, nheads, headdim, dtype, device, continuous, spread)
            torch.cuda.synchronize()
            start_event.record()
            _ = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False)
            end_event.record()
            torch.cuda.synchronize()
            total_time_packed += start_event.elapsed_time(end_event)
            successful_iters += 1
        except torch.cuda.OutOfMemoryError as e:
            print("Packed QKV Attention OOM error:", e)
            if clear_cache:
                torch.cuda.empty_cache()
            continue
        finally:
            if clear_cache:
                torch.cuda.empty_cache()
    avg_packed = total_time_packed / successful_iters if successful_iters > 0 else float('nan')
    results['Packed QKV Attention'] = avg_packed

    # -----------------------
    # Benchmark 3: KV Cache Attention
    # -----------------------
    # 这里演示利用新参数模拟“chunk”分布的情况，只有在非连续模式下才使用 chunk 模式。
    if not continuous and config.get("use_chunk", False):
        # 使用参数 init_text_length、num_chunks 来生成 k,v cache。注意 seqlen_k 应该与 init_text_length 一致
        q = generate_tensor(batch_size, seqlen_q, nheads, headdim, dtype, device, continuous, spread)
        # 使用 chunk 模拟非连续内存布局：
        k_cache = generate_tensor_by_chunks(batch_size, seqlen_k, nheads, headdim, config["num_chunks"], dtype, device)
        v_cache = generate_tensor_by_chunks(batch_size, seqlen_k, nheads, headdim, config["num_chunks"], dtype, device)
    else:
        q = generate_tensor(batch_size, seqlen_q, nheads, headdim, dtype, device, continuous, spread)
        k_cache = generate_tensor(batch_size, seqlen_k, nheads, headdim, dtype, device, continuous, spread)
        v_cache = generate_tensor(batch_size, seqlen_k, nheads, headdim, dtype, device, continuous, spread)

    torch.cuda.synchronize()
    start_event.record()
    _ = flash_attn_with_kvcache(q, k_cache, v_cache, softmax_scale=None, causal=True)
    end_event.record()
    torch.cuda.synchronize()
    warmup_time_cache = start_event.elapsed_time(end_event)
    warmup_times['KV Cache Attention'] = warmup_time_cache
    if clear_cache:
        torch.cuda.empty_cache()

    total_time_cache = 0.0
    successful_iters = 0
    for _ in tqdm(range(num_iters), desc="KV Cache Attention iterations", leave=False):
        try:
            if not continuous and config.get("use_chunk", False):
                q = generate_tensor(batch_size, seqlen_q, nheads, headdim, dtype, device, continuous, spread)
                k_cache = generate_tensor_by_chunks(batch_size, seqlen_k, nheads, headdim, config["num_chunks"], dtype, device)
                v_cache = generate_tensor_by_chunks(batch_size, seqlen_k, nheads, headdim, config["num_chunks"], dtype, device)
            else:
                q = generate_tensor(batch_size, seqlen_q, nheads, headdim, dtype, device, continuous, spread)
                k_cache = generate_tensor(batch_size, seqlen_k, nheads, headdim, dtype, device, continuous, spread)
                v_cache = generate_tensor(batch_size, seqlen_k, nheads, headdim, dtype, device, continuous, spread)
            torch.cuda.synchronize()
            start_event.record()
            _ = flash_attn_with_kvcache(q, k_cache, v_cache, softmax_scale=None, causal=True)
            end_event.record()
            torch.cuda.synchronize()
            total_time_cache += start_event.elapsed_time(end_event)
            successful_iters += 1
        except torch.cuda.OutOfMemoryError as e:
            print("KV Cache Attention OOM error:", e)
            if clear_cache:
                torch.cuda.empty_cache()
            continue
        finally:
            if clear_cache:
                torch.cuda.empty_cache()
    avg_cache = total_time_cache / successful_iters if successful_iters > 0 else float('nan')
    results['KV Cache Attention'] = avg_cache

    print("Warmup times (ms):", warmup_times)
    print("Average times (ms):", results)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flash Attention Benchmark")
    parser.add_argument("--iters", type=int, default=100, help="迭代次数")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--seqlen-q", type=int, default=1024, help="Query 序列长度")
    parser.add_argument("--seqlen-k", type=int, default=1024, help="Key/Value 序列长度")
    parser.add_argument("--nheads", type=int, default=12, help="Attention heads 数量")
    parser.add_argument("--headdim", type=int, default=64, help="每个 head 的维度")
    parser.add_argument("--spread", type=int, default=1, help="非连续内存分布的间隔")
    parser.add_argument("--continuous", action="store_true", help="是否使用连续内存布局")
    parser.add_argument("--use-chunk", action="store_true", help="是否对 KV cache 使用 chunk 模式")
    parser.add_argument("--num-chunks", type=int, default=4, help="KV cache 分块数量")
    parser.add_argument("--init-text-length", type=int, default=1024, help="初始文本长度（用于 KV cache 模拟）")
    parser.add_argument("--chunk-kv", type=int, default=256, help="每个 chunk 中 KV 的数量（用于 KV cache 模拟）")
    parser.add_argument("--clear-cache", type=lambda x: (str(x).lower() == 'true'), default=True, help="是否在每次迭代后清空缓存")
    
    args = parser.parse_args()
    
    # 构建基础配置字典
    config = {
        "batch_size": args.batch_size,
        "seqlen_q": args.seqlen_q,
        "seqlen_k": args.seqlen_k,
        "nheads": args.nheads,
        "headdim": args.headdim,
        "name": "Flash Attention Benchmark",
        "use_chunk": args.use_chunk,
    }
    
    # 针对不同内存布局 configs，分别执行 benchmark
    print("Running benchmark for continuous memory layout...")
    run_one_benchmark_config(config, continuous=True, spread=args.spread, num_iters=args.iters, clear_cache=args.clear_cache)
    
    print("\nRunning benchmark for non-continuous memory layout...")
    run_one_benchmark_config(config, continuous=False, spread=args.spread, num_iters=args.iters, clear_cache=args.clear_cache)
