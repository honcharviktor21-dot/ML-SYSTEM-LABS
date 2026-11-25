import os
import json
import yaml
import torch
import torch.nn as nn
from src.models import (
    set_seed,
    get_device,
    TeacherCNN,
    count_parameters,
    get_model_disk_size_mb,
    get_loaders,
    evaluate_accuracy,
)
from src.pruning import run_pruning_experiments
from src.distillation import run_distillation_study
from src.benchmark import run_latency_and_storage_benchmark, measure_inference_latency_ms


def train_teacher(config, train_loader, test_loader, device):
    print("--- Task 1: Training Oversized Teacher Baseline (~1M Params) ---")
    set_seed(config["experiment"]["seed"])
    t_cfg = config["teacher"]
    teacher = TeacherCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=t_cfg["lr"])

    for epoch in range(1, t_cfg["epochs"] + 1):
        teacher.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(teacher(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        acc = evaluate_accuracy(teacher, test_loader, device)
        print(f"  Epoch {epoch}/{t_cfg['epochs']} | Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}")

    torch.save(teacher.state_dict(), t_cfg["checkpoint_path"])
    params = count_parameters(teacher)
    size_mb = get_model_disk_size_mb(teacher)
    final_acc = evaluate_accuracy(teacher, test_loader, device)
    bs1_ms = measure_inference_latency_ms(teacher, device, batch_size=1)

    print(f"\n  Teacher Summary -> Params: {params:,} | Disk Size: {size_mb} MB | Acc: {final_acc:.4f} | BS=1 Latency: {bs1_ms:.3f} ms\n")
    return teacher, {
        "params": params,
        "disk_size_mb": size_mb,
        "accuracy": final_acc,
        "bs1_latency_ms": bs1_ms,
    }


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"=== Launching Lab 7 Compression Pipeline on {device} ===\n")

    train_loader, test_loader = get_loaders(config)

    # Task 1: Oversized Teacher
    teacher, teacher_summary = train_teacher(config, train_loader, test_loader, device)

    # Task 2: Global, Layer-Wise & Structured Pruning
    pruning_summary, saved_pruned_models = run_pruning_experiments(teacher, train_loader, test_loader, config, device)

    # Task 4: Knowledge Distillation
    distill_summary, plain_student, distilled_student = run_distillation_study(teacher, train_loader, test_loader, config, device)

    # Task 3: Latency & Dense vs. Sparse Storage Benchmark
    bench_summary = run_latency_and_storage_benchmark(
        teacher, saved_pruned_models, plain_student, distilled_student, config, device
    )

    full_summary = {
        "teacher_baseline": teacher_summary,
        "pruning_experiments": pruning_summary,
        "distillation_study": distill_summary,
        "latency_and_storage": bench_summary,
    }

    with open("results/lab7_summary.json", "w") as f:
        json.dump(full_summary, f, indent=4)

    print("Lab 7 Execution Complete. Summary, plots, and model files saved to results/.")