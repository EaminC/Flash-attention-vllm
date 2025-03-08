#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func, flash_attn_with_kvcache
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def benchmark_flash_attention(spread_min, spread_max, spread_step, batch_size, seqlen_q, seqlen_k, nheads, headdim):
    """
    运行 Flash Attention 的基准测试
      spread_min, spread_max, spread_step: 用于模拟不连续内存时，K/V采样的数量范围
      batch_size: 批大小
      seqlen_q: Q 序列的长度
      seqlen_k: K/V 序列的长度
      nheads: 注意力头数
      headdim: 每个头的维度
    返回：
      pandas.DataFrame, 包含不同 spread_k 下的计算时间和差异数据
    """
    results = []
    
    # 确保 spread_max 不超过 seqlen_k
    spread_max = min(spread_max, seqlen_k)
    
    for spread_k in tqdm(range(spread_min, spread_max + 1, spread_step)):
        # 生成 Q, K, V 张量 (连续内存)
        q = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=torch.float16).cuda()
        k = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=torch.float16).cuda()
        v = torch.randn(batch_size, seqlen_k, nheads, headdim, dtype=torch.float16).cuda()

        # 1. 连续内存情况下的 Flash Attention 计算
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        output_continuous = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
        end_event.record()
        torch.cuda.synchronize()
        time_continuous = start_event.elapsed_time(end_event)

        # 2. 不连续内存情况的 Flash Attention 计算
        # 这里通过随机采样 spread_k 个索引来模拟内存不连续的情况
        spread_indices = np.random.choice(seqlen_k, spread_k, replace=False)
        k_spread = k[:, spread_indices, :, :]
        v_spread = v[:, spread_indices, :, :]
        
        # 注意这里重新使用 cuda.Event 进行计时，确保两个测试的计时互不影响
        start_event2 = torch.cuda.Event(enable_timing=True)
        end_event2 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event2.record()
        output_discontinuous = flash_attn_func(q, k_spread, v_spread, dropout_p=0.0, softmax_scale=None, causal=False)
        end_event2.record()
        torch.cuda.synchronize()
        time_discontinuous = start_event2.elapsed_time(end_event2)

        # 记录结果
        results.append({
            'spread_k': spread_k,
            'time_continuous': time_continuous,
            'time_discontinuous': time_discontinuous,
            'abs_diff': time_continuous - time_discontinuous,
            'rel_diff': (time_continuous - time_discontinuous) / time_continuous * 100 if time_continuous != 0 else 0
        })

        # 释放内存，并清空 CUDA 缓存
        del q, k, v, output_continuous, output_discontinuous, k_spread, v_spread
        torch.cuda.empty_cache()
        
    return pd.DataFrame(results)

def plot_results(df):
    """
    使用 matplotlib 作图，比较连续与不连续内存下 Flash Attention 的计算时间
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df['spread_k'], df['time_continuous'], label='连续内存', linestyle='-', color='b', marker='o')
    plt.plot(df['spread_k'], df['time_discontinuous'], label='不连续内存', linestyle='--', color='r', marker='x')
    plt.xlabel('抽样数量 spread_k')
    plt.ylabel('时间 (ms)')
    plt.title('Flash Attention 性能基准测试')
    plt.legend()
    plt.grid(True)
    plt.savefig('flash_attention_benchmark.png')
    plt.show()

def main():
    # 参数设置
    batch_size = 2
    seqlen_q = 4
    seqlen_k = 4   # 注意：这里必须保证 seqlen_k >= spread_max
    nheads = 8
    headdim = 64
    spread_min = 1
    spread_max = 4  # 设置不连续抽样上限为4（不能超过 seqlen_k）
    spread_step = 1

    # 运行基准测试，得到结果 DataFrame
    results_df = benchmark_flash_attention(spread_min, spread_max, spread_step, batch_size, seqlen_q, seqlen_k, nheads, headdim)
    print(results_df)
    plot_results(results_df)

if __name__ == "__main__":
    main()
