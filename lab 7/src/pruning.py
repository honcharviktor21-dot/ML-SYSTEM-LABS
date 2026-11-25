import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.models import evaluate_accuracy


def get_prunable_weights(model: nn.Module):
    """Returns dict of {param_name: param_tensor} for Conv2d and Linear weights."""
    prunable = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prunable[f"{name}.weight"] = module.weight
    return prunable


def compute_global_masks(model: nn.Module, sparsity: float):
    prunable = get_prunable_weights(model)
    if sparsity <= 0.0:
        return {k: torch.ones_like(w) for k, w in prunable.items()}

    all_weights = torch.cat([w.detach().abs().view(-1) for w in prunable.values()])
    k = int(sparsity * all_weights.numel())
    if k == 0:
        return {name: torch.ones_like(w) for name, w in prunable.items()}

    threshold = torch.kthvalue(all_weights, k).values
    masks = {}
    for name, w in prunable.items():
        masks[name] = (w.detach().abs() > threshold).float()
    return masks


def compute_layerwise_masks(model: nn.Module, sparsity: float):
    prunable = get_prunable_weights(model)
    masks = {}
    for name, w in prunable.items():
        if sparsity <= 0.0:
            masks[name] = torch.ones_like(w)
            continue
        flat = w.detach().abs().view(-1)
        k = int(sparsity * flat.numel())
        if k == 0:
            masks[name] = torch.ones_like(w)
        else:
            threshold = torch.kthvalue(flat, k).values
            masks[name] = (w.detach().abs() > threshold).float()
    return masks


def compute_structured_l2_masks(model: nn.Module, sparsity: float):
    """Removes whole Conv2d filters ranked by L2 norm."""
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            w = module.weight.detach()
            # L2 norm per output filter: shape [out_channels]
            filter_norms = torch.norm(w.view(w.size(0), -1), p=2, dim=1)
            num_prune = int(sparsity * w.size(0))
            mask = torch.ones_like(w)
            if num_prune > 0:
                prune_idx = torch.argsort(filter_norms)[:num_prune]
                mask[prune_idx] = 0.0
            masks[f"{name}.weight"] = mask
    return masks


def apply_masks(model: nn.Module, masks: dict):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name])


def measure_actual_sparsity(model: nn.Module) -> float:
    prunable = get_prunable_weights(model)
    total = sum(w.numel() for w in prunable.values())
    zeros = sum((w == 0).sum().item() for w in prunable.values())
    return round(zeros / total, 4)


def finetune_with_fixed_mask(model: nn.Module, masks: dict, train_loader, epochs: int, lr: float, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()

            # Zero out gradients of masked weights so they never update
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in masks and param.grad is not None:
                        param.grad.mul_(masks[name])

            optimizer.step()
            apply_masks(model, masks)


def run_pruning_experiments(base_teacher: nn.Module, train_loader, test_loader, config, device):
    print("--- Task 2: Magnitude Pruning (Global, Layer-Wise & Structured L2) ---")
    p_cfg = config["pruning"]
    unstructured_targets = p_cfg["unstructured_sparsities"]
    structured_targets = p_cfg["structured_sparsities"]
    ft_epochs = p_cfg["finetune_epochs"]
    ft_lr = p_cfg["finetune_lr"]

    global_results = []
    layerwise_results = []
    structured_results = []
    saved_models = {}

    # 1. Global Unstructured Pruning
    print("  [1/3] Global Unstructured Pruning:")
    for sp in unstructured_targets:
        model = copy.deepcopy(base_teacher).to(device)
        masks = compute_global_masks(model, sp)
        apply_masks(model, masks)
        if sp > 0.0:
            finetune_with_fixed_mask(model, masks, train_loader, ft_epochs, ft_lr, device)
        acc = evaluate_accuracy(model, test_loader, device)
        actual_sp = measure_actual_sparsity(model)
        global_results.append({"target_sparsity": sp, "actual_sparsity": actual_sp, "accuracy": acc})
        print(f"    Target={sp*100:5.1f}% | Actual={actual_sp*100:5.1f}% | Accuracy={acc:.4f}")
        if sp in (0.90, 0.95):
            saved_models[f"global_{int(sp*100)}"] = copy.deepcopy(model)

    # 2. Layer-Wise Unstructured Pruning
    print("  [2/3] Layer-Wise Unstructured Pruning:")
    for sp in unstructured_targets:
        model = copy.deepcopy(base_teacher).to(device)
        masks = compute_layerwise_masks(model, sp)
        apply_masks(model, masks)
        if sp > 0.0:
            finetune_with_fixed_mask(model, masks, train_loader, ft_epochs, ft_lr, device)
        acc = evaluate_accuracy(model, test_loader, device)
        actual_sp = measure_actual_sparsity(model)
        layerwise_results.append({"target_sparsity": sp, "actual_sparsity": actual_sp, "accuracy": acc})
        print(f"    Target={sp*100:5.1f}% | Actual={actual_sp*100:5.1f}% | Accuracy={acc:.4f}")

    # 3. Structured L2 Filter Pruning
    print("  [3/3] Structured L2 Filter Pruning:")
    for sp in structured_targets:
        model = copy.deepcopy(base_teacher).to(device)
        masks = compute_structured_l2_masks(model, sp)
        apply_masks(model, masks)
        finetune_with_fixed_mask(model, masks, train_loader, ft_epochs, ft_lr, device)
        acc = evaluate_accuracy(model, test_loader, device)
        actual_sp = measure_actual_sparsity(model)
        structured_results.append({"filter_sparsity": sp, "overall_weight_sparsity": actual_sp, "accuracy": acc})
        print(f"    Filter Sparsity={sp*100:5.1f}% | Overall Weight Sparsity={actual_sp*100:5.1f}% | Accuracy={acc:.4f}")
        if sp == 0.50:
            saved_models["structured_50"] = copy.deepcopy(model)
    print()

    # Plot Accuracy vs Sparsity for all three pruning regimes
    plt.figure(figsize=(8.5, 5))
    plt.plot(
        [r["target_sparsity"] * 100 for r in global_results],
        [r["accuracy"] * 100 for r in global_results],
        marker="o", linewidth=2, label="Unstructured Global Pruning"
    )
    plt.plot(
        [r["target_sparsity"] * 100 for r in layerwise_results],
        [r["accuracy"] * 100 for r in layerwise_results],
        marker="s", linewidth=2, linestyle="--", label="Unstructured Layer-Wise Pruning"
    )
    plt.plot(
        [r["filter_sparsity"] * 100 for r in structured_results],
        [r["accuracy"] * 100 for r in structured_results],
        marker="^", linewidth=2.5, color="crimson", label="Structured L2 Filter Pruning"
    )
    plt.title("Fashion-MNIST Accuracy vs. Pruning Sparsity (After 3-Epoch Fine-Tuning)")
    plt.xlabel("Target Sparsity (%)")
    plt.ylabel("Top-1 Test Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/pruning_sparsity_curves.png", dpi=150)
    plt.close()

    return {
        "global_unstructured": global_results,
        "layerwise_unstructured": layerwise_results,
        "structured_l2": structured_results,
    }, saved_models