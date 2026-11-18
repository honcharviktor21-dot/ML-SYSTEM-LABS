import time
import tracemalloc
import torch
import torch.nn as nn
from src.model import set_seed, get_device, build_model, get_loaders, evaluate_accuracy


def run_batch_study(config):
    print("--- Task 4: Batch-Size Study (5 Epochs) ---")
    device = get_device()
    b_cfg = config["batch_study"]
    epochs = b_cfg["epochs"]
    lr = b_cfg["lr"]

    results = []
    for bs in b_cfg["batch_sizes"]:
        set_seed(config["experiment"]["seed"])
        train_loader, test_loader = get_loaders(config, batch_size=bs, num_workers=0)
        model = build_model(config, device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        else:
            tracemalloc.start()

        total_samples = 0
        t0 = time.perf_counter()
        for _ in range(epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                total_samples += x.size(0)
        elapsed = time.perf_counter() - t0

        if torch.cuda.is_available():
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mem_mb = peak_bytes / (1024 ** 2)

        acc = evaluate_accuracy(model, test_loader, device)
        throughput = total_samples / elapsed

        row = {
            "batch_size": bs,
            "throughput_samples_sec": round(throughput, 1),
            "peak_memory_mb": round(peak_mem_mb, 2),
            "final_accuracy": round(acc, 4),
        }
        results.append(row)
        print(f"  BS={bs:<3} | Throughput={row['throughput_samples_sec']:7.1f} img/s | PeakMem={row['peak_memory_mb']:6.2f} MB | Acc={row['final_accuracy']:.4f}")
    print()
    return results