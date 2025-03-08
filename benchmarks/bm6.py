import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 生成离散的 K/V，优化内存分配
def generate_discontinuous_kv(batch_size, seq_len, nheads, headdim, spread_k):
    segment_len = seq_len // spread_k  # 每段基础长度
    remainder = seq_len % spread_k  # 计算剩余 token
    
    k = torch.empty(batch_size, seq_len, nheads, headdim, device="cuda")
    v = torch.empty(batch_size, seq_len, nheads, headdim, device="cuda")
    
    start_idx = 0
    for i in range(spread_k):
        cur_len = segment_len + (1 if i < remainder else 0)  # 前 remainder 段多 1 个 token
        k[:, start_idx:start_idx+cur_len, :, :] = torch.randn(batch_size, cur_len, nheads, headdim, device="cuda")
        v[:, start_idx:start_idx+cur_len, :, :] = torch.randn(batch_size, cur_len, nheads, headdim, device="cuda")
        start_idx += cur_len
    
    return k, v

# 生成连续的 K/V
def generate_continuous_kv(batch_size, seq_len, nheads, headdim):
    k = torch.randn(batch_size, seq_len, nheads, headdim, device="cuda")
    v = torch.randn(batch_size, seq_len, nheads, headdim, device="cuda")
    return k, v

# 更精确的 Flash Attention 计算（优化计时方式）
def flash_attention(q, k, v):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    output = torch.matmul(q, k.transpose(-2, -1)) @ v  # 注意力计算
    end_event.record()
    
    torch.cuda.synchronize()  # 确保所有 CUDA 操作完成
    return start_event.elapsed_time(end_event) / 1000.0  # 转换为秒

# 运行基准测试
def benchmark_flash_attention(batch_size, seq_len, nheads, headdim, spread_min, spread_max, spread_step):
    results = []

    # 热身运行，避免首个数据点异常
    print("Running warm-up...")
    q = torch.empty(batch_size, seq_len, nheads, headdim, device="cuda")
    torch.nn.init.xavier_uniform_(q)  # 更合理的初始化方式
    k, v = generate_continuous_kv(batch_size, seq_len, nheads, headdim)
    flash_attention(q, k, v)

    # 遍历不同的 spread_k 取值
    spread_k_values = list(range(spread_min, spread_max + 1, spread_step))
    for spread_k in tqdm(spread_k_values, desc="Benchmarking spread_k values"):
        # 生成 Query
        q = torch.empty(batch_size, seq_len, nheads, headdim, device="cuda")
        torch.nn.init.xavier_uniform_(q)  # 更接近 Transformer 真实初始化

        # 生成不同内存布局的 K/V
        k_disc, v_disc = generate_discontinuous_kv(batch_size, seq_len, nheads, headdim, spread_k)
        k_cont, v_cont = generate_continuous_kv(batch_size, seq_len, nheads, headdim)

        # 运行 Flash Attention
        time_disc = flash_attention(q, k_disc, v_disc)
        time_cont = flash_attention(q, k_cont, v_cont)

        results.append(("flash_attention", spread_k, time_disc, time_cont))

        # 释放 GPU 内存
        del q, k_disc, v_disc, k_cont, v_cont
        torch.cuda.empty_cache()

    # 画图
    plot_results(results, spread_k_values)

# 绘制基准测试结果
def plot_results(results, spread_k_values):
    methods = set([r[0] for r in results])
    colors = {"flash_attention": "blue"}

    plt.figure(figsize=(10, 6))

    for method in methods:
        x_vals = []
        y_disc = []
        y_cont = []
        for spread_k in spread_k_values:
            for r in results:
                if r[0] == method and r[1] == spread_k:
                    x_vals.append(spread_k)
                    y_disc.append(r[2])
                    y_cont.append(r[3])

        color = colors.get(method, "gray")
        plt.plot(x_vals, y_disc, linestyle="dashed", marker="o", color=color, label=f"{method} (Discontinuous)")
        plt.plot(x_vals, y_cont, linestyle="solid", marker="o", color=color, label=f"{method} (Continuous)")

    plt.xlabel("spread_k")
    plt.ylabel("Execution Time (s)")
    plt.title("Flash Attention Performance")
    plt.legend()
    plt.grid(True)
    plt.savefig("flash_attention_benchmark.png")
    plt.show()

# 运行基准测试
if __name__ == "__main__":
    benchmark_flash_attention(batch_size=8, seq_len=4096, nheads=16, headdim=1024, spread_min=1, spread_max=200, spread_step=1)
  