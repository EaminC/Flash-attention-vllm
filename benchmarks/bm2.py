# -*- coding: utf-8 -*-
"""
This program benchmarks three Flash Attention implementations 
(Basic Attention, Packed QKV Attention, and KV Cache Attention)
under two sets of parameters and generates two PNG images.

It clears the CUDA cache after each atomic operation by calling torch.cuda.empty_cache(),
and uses tqdm progress bars to visually display progress.

Usage example:
   python bm2.py --iters 100 --spread_min 1 --spread_max 10 --spread_step 1
"""

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Import Flash Attention functions.
# Make sure that the flash_attn module is added to the PYTHON path.
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func, flash_attn_with_kvcache

def generate_tensor(batch_size, seqlen, nheads, headdim, dtype=torch.float16, device='cuda', continuous=True, spread=1):
    """
    Generates a tensor for flash attention.
    If continuous is True, returns a tensor with contiguous memory.
    If False, generates a longer tensor (length = seqlen * spread) and slices it to return a non-contiguous tensor.
    """
    if continuous or spread <= 1:
        return torch.randn(batch_size, seqlen, nheads, headdim, dtype=dtype, device=device)
    else:
        full_tensor = torch.randn(batch_size, seqlen * spread, nheads, headdim, dtype=dtype, device=device)
        return full_tensor[:, ::spread, :, :]

def generate_qkv_tensor(batch_size, seqlen, nheads, headdim, dtype=torch.float16, device='cuda', continuous=True, spread=1):
    """
    Generates a packed QKV tensor with shape (B, seqlen, 3, nheads, headdim).
    Supports both contiguous and non-contiguous memory layouts.
    """
    if continuous or spread <= 1:
        return torch.randn(batch_size, seqlen, 3, nheads, headdim, dtype=dtype, device=device)
    else:
        full_tensor = torch.randn(batch_size, seqlen * spread, 3, nheads, headdim, dtype=dtype, device=device)
        return full_tensor[:, ::spread, :, :, :]

def run_one_benchmark_config(config, continuous, spread, num_iters=100):
    """
    Runs benchmarks for three Flash Attention tests for the given configuration
    and calculates the average kernel execution time (in milliseconds).

    Parameters:
        config     : A dictionary containing configuration info (must include batch_size, seqlen_q, seqlen_k, nheads, headdim).
        continuous : boolean flag; True to generate tensors with contiguous memory, False for non-contiguous tensors.
        spread     : The interval parameter used when generating non-contiguous tensors.
        num_iters  : The number of iterations used for timing (warm-up iterations are not counted).

    Returns:
        results      : A dictionary mapping each test name to its average kernel execution time (in ms).
        warmup_times : A dictionary mapping each test name to its warm-up execution time (in ms).
    """
    batch_size = config['batch_size']
    seqlen_q   = config['seqlen_q']
    seqlen_k   = config['seqlen_k']
    nheads     = config['nheads']
    headdim    = config['headdim']
    dtype      = config.get('dtype', torch.float16)
    device     = config.get('device', 'cuda')
    
    layout_str = "Contiguous" if continuous else f"Non-Contiguous (spread={spread})"
    print(f">> Testing {layout_str} layout, configuration: {config.get('name', 'Unknown')}")
    print("--------------------------------------------")
    
    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)
    
    results = {}
    warmup_times = {}
    
    # -----------------------
    # Benchmark 1: Basic Attention
    # -----------------------
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
            torch.cuda.empty_cache()
            continue
        finally:
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
            torch.cuda.empty_cache()
            continue
        finally:
            torch.cuda.empty_cache()
    avg_packed = total_time_packed / successful_iters if successful_iters > 0 else float('nan')
    results['Packed QKV Attention'] = avg_packed

    # -----------------------
    # Benchmark 3: KV Cache Attention
    # -----------------------
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
    torch.cuda.empty_cache()
    
    total_time_cache = 0.0
    successful_iters = 0
    for _ in tqdm(range(num_iters), desc="KV Cache Attention iterations", leave=False):
        try:
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
            torch.cuda.empty_cache()
            continue
        finally:
            torch.cuda.empty_cache()
    avg_cache = total_time_cache / successful_iters if successful_iters > 0 else float('nan')
    results['KV Cache Attention'] = avg_cache

    # Print the results
    header = "{:<25s} | {:>20s} | {:>20s}".format("Benchmark", "Warmup Time (ms)", "Average Kernel Time (ms)")
    divider = "-" * len(header)
    print("\n" + divider)
    print(header)
    print(divider)
    for key in results:
        print("{:<25s} | {:>20.3f} | {:>20.3f}".format(key, warmup_times[key], results[key]))
    print(divider + "\n")
    
    # Clear cache and return results
    torch.cuda.empty_cache()
    return results, warmup_times

def collect_benchmarks_for_config(config, spreads, num_iters):
    """
    Collects benchmark results for the given configuration in non-contiguous mode (across different spread values).
    Returns a dictionary mapping each test name to a list of average kernel times (one for each spread value).
    """
    bench_results = {
        "Basic Attention": [],
        "Packed QKV Attention": [],
        "KV Cache Attention": []
    }
    
    for s in tqdm(spreads, desc="Spread iterations"):
        results, _ = run_one_benchmark_config(config, continuous=False, spread=s, num_iters=num_iters)
        for bench in bench_results:
            bench_results[bench].append(results[bench])
        # Clear cache to prevent memory overflow
        torch.cuda.empty_cache()
        
    return bench_results

def plot_results_for_config(config_name, spreads, contiguous_results, noncontiguous_results):
    """
    Plots the benchmark results for the given configuration.
    Each test displays two curves: one for contiguous mode and one for non-contiguous mode.

    Parameters:
        config_name           : Name of the configuration, e.g., "Basic" or "BERT Base".
        spreads               : List of spread parameter values (x-axis).
        contiguous_results    : A dictionary mapping each test to its contiguous mode average time (scalar).
        noncontiguous_results : A dictionary mapping each test to its non-contiguous mode average times (list).
    """
    plt.figure(figsize=(10, 7))
    
    # Define colors for each test
    colors = {
        "Basic Attention": "blue",
        "Packed QKV Attention": "orange",
        "KV Cache Attention": "green"
    }
    
    # Define line styles and markers for contiguous and non-contiguous modes
    line_styles = {
        "Contiguous": {"linestyle": "-", "marker": "o"},
        "Non-Contiguous": {"linestyle": "--", "marker": "x"}
    }
    
    for bench in contiguous_results:
        # Plot contiguous result as a horizontal line
        plt.plot(spreads, [contiguous_results[bench]] * len(spreads),
                 color=colors[bench],
                 label=f"{bench} (Contiguous)",
                 **line_styles["Contiguous"])
        # Plot non-contiguous results
        plt.plot(spreads, noncontiguous_results[bench],
                 color=colors[bench],
                 label=f"{bench} (Non-Contiguous)",
                 **line_styles["Non-Contiguous"])
    
    plt.xlabel("Spread Parameter")
    plt.ylabel("Average Kernel Time (ms)")
    plt.title(f"Flash Attention Benchmark Results - {config_name}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{config_name}_flash_attention_benchmark.png")
    plt.close()

# Additional code (such as argument parsing and main() function) would go here.
# This snippet focuses solely on the benchmarking functions and their implementations.
