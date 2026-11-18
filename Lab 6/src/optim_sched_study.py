import torch
import torch.nn as nn
from src.model import set_seed, get_device, build_model, get_loaders, evaluate_accuracy


def train_with_combo(config, opt_name, sched_name, device):
    set_seed(config["experiment"]["seed"])
    o_cfg = config["optim_sched_study"]
    epochs = o_cfg["epochs"]
    train_loader, test_loader = get_loaders(config, batch_size=o_cfg["batch_size"], num_workers=0)
    model = build_model(config, device)
    criterion = nn.CrossEntropyLoss()

    if opt_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=o_cfg["lr_adam"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=o_cfg["lr_sgd"], momentum=o_cfg["momentum"])

    if sched_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    elif sched_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    epoch_accs = []
    for _ in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_loss)
        else:
            scheduler.step()

        epoch_accs.append(round(evaluate_accuracy(model, test_loader, device), 4))

    return epoch_accs


def run_optim_sched_study(config):
    print("--- Task 5: Optimizers & LR Schedulers (10 Epochs) ---")
    device = get_device()
    combos = [
        ("Adam", "StepLR"),
        ("Adam", "CosineAnnealingLR"),
        ("Adam", "ReduceLROnPlateau"),
        ("SGD+Momentum", "StepLR"),
        ("SGD+Momentum", "CosineAnnealingLR"),
        ("SGD+Momentum", "ReduceLROnPlateau"),
    ]

    results = []
    for opt_name, sched_name in combos:
        accs = train_with_combo(config, opt_name, sched_name, device)
        row = {
            "optimizer": opt_name,
            "scheduler": sched_name,
            "epoch_5_acc": accs[4],
            "final_epoch_10_acc": accs[-1],
        }
        results.append(row)
        print(f"  {opt_name:<12} + {sched_name:<18} -> Epoch 5 Acc: {row['epoch_5_acc']:.4f} | Final Acc: {row['final_epoch_10_acc']:.4f}")
    print()
    return results