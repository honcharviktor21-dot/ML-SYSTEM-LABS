import time
import statistics
import json
import torch
import torch.nn as nn

def benchmark_pytorch_execution(config):
    print("--- Task 6: Execution Modes & Memory Contiguity ---")
    size = config['execution_benchmark']['matrix_size']
    warmup = config['execution_benchmark']['warmup_iters']
    iters = config['execution_benchmark']['benchmark_iters']

    class BenchmarkModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(size, size),
                nn.ReLU(),
                nn.Linear(size, size)
            )
        def forward(self, x):
            return self.net(x)

    model = BenchmarkModel().eval()
    x = torch.randn(size, size)

    # 1. Eager Mode
    for _ in range(warmup):
        _ = model(x)
    eager_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model(x)
        eager_times.append(time.perf_counter() - t0)
    eager_median = statistics.median(eager_times)

    # 2. Traced Execution (TorchScript JIT)
    traced_model = torch.jit.trace(model, x)
    for _ in range(warmup):
        _ = traced_model(x)
    traced_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = traced_model(x)
        traced_times.append(time.perf_counter() - t0)
    traced_median = statistics.median(traced_times)

    # 3. Memory Contiguity Benchmark
    mat = torch.randn(3000, 3000)
    mat_non_contig = mat.t()  # Transposed view (strided, non-contiguous)

    print(f"Contiguous Tensor Shape: {mat.shape}, Stride: {mat.stride()}, Contiguous: {mat.is_contiguous()}")
    print(f"Transposed Tensor Shape: {mat_non_contig.shape}, Stride: {mat_non_contig.stride()}, Contiguous: {mat_non_contig.is_contiguous()}")

    for _ in range(warmup):
        _ = torch.sum(mat)
        _ = torch.sum(mat_non_contig)

    contig_times = []
    non_contig_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = torch.sum(mat)
        contig_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = torch.sum(mat_non_contig)
        non_contig_times.append(time.perf_counter() - t0)

    contig_median = statistics.median(contig_times)
    non_contig_median = statistics.median(non_contig_times)

    print(f"\nEager Execution Median Time:  {eager_median * 1000:.3f} ms")
    print(f"Traced Execution Median Time: {traced_median * 1000:.3f} ms")
    print(f"Contiguous Sum Median Time:     {contig_median * 1000:.3f} ms")
    print(f"Non-Contiguous Sum Median Time: {non_contig_median * 1000:.3f} ms\n")

    bench_results = {
        "eager_median_ms": eager_median * 1000,
        "traced_median_ms": traced_median * 1000,
        "contiguous_sum_ms": contig_median * 1000,
        "non_contiguous_sum_ms": non_contig_median * 1000,
        "contiguity_overhead_ratio": non_contig_median / contig_median
    }

    with open("results/benchmark_results.json", "w") as f:
        json.dump(bench_results, f, indent=4)

    return bench_results