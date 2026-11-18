import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.model import set_seed, get_device, build_model, get_loaders, evaluate_accuracy


def run_grad_accum_study(config):
    print("--- Task 6: Gradient Accumulation (Batch 256 vs 64 x 4) ---")
    device = get_device()
    g_cfg = config["grad_accum"]
    epochs = g_cfg["epochs"]
    lr = g_cfg["lr"]

    # 1. Native Batch 256
    set_seed(config["experiment"]["seed"])
    train_256, test_loader = get_loaders(config, batch_size=g_cfg["large_batch"], num_workers=0)
    model_256 = build_model(config, device)
    opt_256 = torch.optim.Adam(model_256.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses_256 = []
    for _ in range(epochs):
        model_256.train()
        total_loss = 0.0
        for x, y in train_256:
            x, y = x.to(device), y.to(device)
            opt_256.zero_grad()
            loss = criterion(model_256(x), y)
            loss.backward()
            opt_256.step()
            total_loss += loss.item() * x.size(0)
        losses_256.append(round(total_loss / len(train_256.dataset), 4))
    acc_256 = round(evaluate_accuracy(model_256, test_loader, device), 4)

    # 2. Micro-batch 64 x 4 Accumulation Steps
    set_seed(config["experiment"]["seed"])
    train_64, _ = get_loaders(config, batch_size=g_cfg["micro_batch"], num_workers=0)
    model_acc = build_model(config, device)
    opt_acc = torch.optim.Adam(model_acc.parameters(), lr=lr)
    accum_steps = g_cfg["accum_steps"]

    losses_acc = []
    for _ in range(epochs):
        model_acc.train()
        total_loss = 0.0
        opt_acc.zero_grad()
        for step, (x, y) in enumerate(train_64):
            x, y = x.to(device), y.to(device)
            raw_loss = criterion(model_acc(x), y)
            scaled_loss = raw_loss / accum_steps
            scaled_loss.backward()
            total_loss += raw_loss.item() * x.size(0)

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_64):
                opt_acc.step()
                opt_acc.zero_grad()
        losses_acc.append(round(total_loss / len(train_64.dataset), 4))
    acc_accum = round(evaluate_accuracy(model_acc, test_loader, device), 4)

    print(f"  Native Batch 256  -> Final Loss: {losses_256[-1]:.4f} | Test Acc: {acc_256:.4f}")
    print(f"  Batch 64 x 4 Acc  -> Final Loss: {losses_acc[-1]:.4f} | Test Acc: {acc_accum:.4f}\n")

    return {
        "native_256": {"epoch_losses": losses_256, "final_acc": acc_256},
        "accum_64x4": {"epoch_losses": losses_acc, "final_acc": acc_accum},
    }


def run_checkpoint_resume_study(config):
    print("--- Task 7: Interrupt at Epoch 3 & Resume from Checkpoint ---")
    set_seed(config["experiment"]["seed"])
    device = get_device()
    c_cfg = config["checkpoint_resume"]
    train_loader, test_loader = get_loaders(config, batch_size=c_cfg["batch_size"], num_workers=0)

    model = build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=c_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    history = []

    # Phase 1: Train epochs 1..3 and simulate interruption
    for epoch in range(1, c_cfg["interrupt_epoch"] + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        scheduler.step()
        avg_loss = running_loss / len(train_loader.dataset)
        acc = evaluate_accuracy(model, test_loader, device)
        history.append({"epoch": epoch, "phase": "before_interrupt", "loss": round(avg_loss, 4), "acc": round(acc, 4)})
        print(f"  [Phase 1] Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    # Save checkpoint at epoch 3
    ckpt_path = c_cfg["checkpoint_path"]
    torch.save({
        "epoch": c_cfg["interrupt_epoch"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, ckpt_path)
    print(f"  --> Training interrupted. Checkpoint saved to {ckpt_path}")

    # Phase 2: Instantiate fresh objects, load checkpoint, resume epochs 4..6
    resumed_model = build_model(config, device)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=c_cfg["lr"])
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=2, gamma=0.8)

    checkpoint = torch.load(ckpt_path, map_location=device)
    resumed_model.load_state_dict(checkpoint["model_state_dict"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    resumed_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1

    for epoch in range(start_epoch, c_cfg["total_epochs"] + 1):
        resumed_model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            resumed_optimizer.zero_grad()
            loss = criterion(resumed_model(x), y)
            loss.backward()
            resumed_optimizer.step()
            running_loss += loss.item() * x.size(0)
        resumed_scheduler.step()
        avg_loss = running_loss / len(train_loader.dataset)
        acc = evaluate_accuracy(resumed_model, test_loader, device)
        history.append({"epoch": epoch, "phase": "resumed", "loss": round(avg_loss, 4), "acc": round(acc, 4)})
        print(f"  [Resumed] Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
    print()

    # Plot continuity curve
    epochs_list = [h["epoch"] for h in history]
    losses_list = [h["loss"] for h in history]
    accs_list = [h["acc"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(epochs_list, losses_list, "b-o", label="Train Loss")
    ax1.axvline(x=c_cfg["interrupt_epoch"], color="red", linestyle="--", label="Checkpoint Interrupt (Epoch 3)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="b")

    ax2 = ax1.twinx()
    ax2.plot(epochs_list, accs_list, "g-s", label="Test Accuracy")
    ax2.set_ylabel("Accuracy", color="g")

    plt.title("Checkpoint Interrupt & Seamless Training Resume")
    fig.tight_layout()
    plt.savefig("results/resume_curves.png", dpi=150)
    plt.close()

    return history