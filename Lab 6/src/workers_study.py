import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.model import set_seed, get_device, build_model, get_loaders


def run_workers_study(config):
    print("--- Task 3: num_workers Bottleneck Study ---")
    device = get_device()
    w_cfg = config["workers_study"]
    worker_counts = w_cfg["worker_counts"]
    batch_size = w_cfg["batch_size"]

    results = []
    for nw in worker_counts:
        set_seed(config["experiment"]["seed"])
        train_loader, _ = get_loaders(config, batch_size=batch_size, num_workers=nw)
        model = build_model(config, device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        t0 = time.perf_counter()
        total_samples = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_samples += x.size(0)
        elapsed = time.perf_counter() - t0
        throughput = total_samples / elapsed

        results.append({
            "num_workers": nw,
            "epoch_time_sec": round(elapsed, 4),
            "throughput_samples_sec": round(throughput, 1),
        })
        print(f"  num_workers={nw}: time={elapsed:.4f}s, throughput={throughput:.1f} samples/s")

    best = min(results, key=lambda r: r["epoch_time_sec"])
    print(f"  Optimal num_workers={best['num_workers']} ({best['epoch_time_sec']}s)\n")

    plt.figure(figsize=(8, 4.5))
    plt.plot([r["num_workers"] for r in results], [r["epoch_time_sec"] for r in results], marker="o", linewidth=2)
    plt.title("Epoch Training Time vs. DataLoader num_workers")
    plt.xlabel("num_workers")
    plt.ylabel("Epoch Time (seconds)")
    plt.xticks(worker_counts)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/num_workers_plot.png", dpi=150)
    plt.close()

    return results