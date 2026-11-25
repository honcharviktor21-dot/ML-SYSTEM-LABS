import os
import time
import statistics
import torch
import torch.nn as nn
from src.models import CompactedTeacherCNN


def measure_inference_latency_ms(model: nn.Module, device, batch_size=1, warmup=30, iters=200) -> float:
    model.eval()
    x = torch.randn(batch_size, 1, 28, 28, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    return round(statistics.median(times), 4)


def compare_dense_vs_sparse_storage(model: nn.Module, tag: str) -> dict:
    """Saves model in standard dense format vs Sparse COO format and compares disk sizes."""
    dense_path = f"results/{tag}_dense.pt"
    sparse_path = f"results/{tag}_sparse.pt"

    # 1. Standard Dense State Dict
    torch.save(model.state_dict(), dense_path)
    dense_mb = os.path.getsize(dense_path) / (1024 ** 2)

    # 2. Sparse COO State Dict (stores only non-zero values + coordinates)
    sparse_state = {}
    for k, v in model.state_dict().items():
        if v.dim() >= 2 and (v == 0).sum().item() > 0:
            sparse_state[k] = v.to_sparse()
        else:
            sparse_state[k] = v

    torch.save(sparse_state, sparse_path)
    sparse_mb = os.path.getsize(sparse_path) / (1024 ** 2)

    compression_ratio = dense_mb / sparse_mb if sparse_mb > 0 else 1.0
    return {
        "dense_size_mb": round(dense_mb, 3),
        "sparse_size_mb": round(sparse_mb, 3),
        "disk_compression_ratio": round(compression_ratio, 2),
    }


def run_latency_and_storage_benchmark(teacher, saved_pruned_models, plain_student, distilled_student, config, device):
    print("--- Task 3: Latency & Dense vs. Sparse Storage Analysis ---")
    l_cfg = config["latency_benchmark"]
    wu, it = l_cfg["warmup_iters"], l_cfg["measure_iters"]

    # Physically compacted CNN matching 50% channel pruning (32 -> 64 -> 80)
    compacted_50 = CompactedTeacherCNN(c1=32, c2=64, fc=80).to(device)

    models_to_time = {
        "Teacher_Dense_1.08M": teacher,
        "Unstructured_Global_90pct_Masked": saved_pruned_models["global_90"],
        "Unstructured_Global_95pct_Masked": saved_pruned_models["global_95"],
        "Structured_50pct_Masked": saved_pruned_models["structured_50"],
        "Structured_50pct_Compacted": compacted_50,
        "Student_Plain_106k": plain_student,
        "Student_Distilled_106k": distilled_student,
    }

    latency_results = {}
    for name, m in models_to_time.items():
        bs1_ms = measure_inference_latency_ms(m, device, batch_size=1, warmup=wu, iters=it)
        bs128_ms = measure_inference_latency_ms(m, device, batch_size=128, warmup=wu, iters=it)
        latency_results[name] = {
            "bs1_latency_ms": bs1_ms,
            "bs128_latency_ms": bs128_ms,
        }
        print(f"  {name:<34} | BS=1: {bs1_ms:6.3f} ms | BS=128: {bs128_ms:6.3f} ms")

    storage_90 = compare_dense_vs_sparse_storage(saved_pruned_models["global_90"], "pruned_90")
    storage_95 = compare_dense_vs_sparse_storage(saved_pruned_models["global_95"], "pruned_95")

    print(f"\n  Storage (90% Sparse): Dense={storage_90['dense_size_mb']} MB vs Sparse COO={storage_90['sparse_size_mb']} MB ({storage_90['disk_compression_ratio']}x smaller)")
    print(f"  Storage (95% Sparse): Dense={storage_95['dense_size_mb']} MB vs Sparse COO={storage_95['sparse_size_mb']} MB ({storage_95['disk_compression_ratio']}x smaller)\n")

    return {
        "latencies": latency_results,
        "storage_90pct": storage_90,
        "storage_95pct": storage_95,
    }