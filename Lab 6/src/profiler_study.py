import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, record_function
from src.model import set_seed, get_device, build_model, get_loaders


def run_profiler_study(config):
    print("--- Task 2: Profiling One Epoch with torch.profiler ---")
    set_seed(config["experiment"]["seed"])
    device = get_device()

    p_cfg = config["profiler"]
    train_loader, _ = get_loaders(config, batch_size=p_cfg["batch_size"], num_workers=p_cfg["num_workers"])
    model = build_model(config, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    model.train()
    data_iter = iter(train_loader)

    with profile(activities=activities, record_shapes=True) as prof:
        while True:
            with record_function("phase_data_loading"):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    break
                x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            with record_function("phase_forward_pass"):
                out = model(x)
                loss = criterion(out, y)

            with record_function("phase_backward_pass"):
                loss.backward()

            with record_function("phase_optimizer_step"):
                optimizer.step()

    phases = {
        "data_loading": 0.0,
        "forward_pass": 0.0,
        "backward_pass": 0.0,
        "optimizer_step": 0.0,
    }

    for evt in prof.key_averages():
        if evt.key == "phase_data_loading":
            phases["data_loading"] += evt.cpu_time_total
        elif evt.key == "phase_forward_pass":
            phases["forward_pass"] += evt.cpu_time_total
        elif evt.key == "phase_backward_pass":
            phases["backward_pass"] += evt.cpu_time_total
        elif evt.key == "phase_optimizer_step":
            phases["optimizer_step"] += evt.cpu_time_total

    total_time = sum(phases.values()) or 1.0
    shares = {k: round((v / total_time) * 100.0, 2) for k, v in phases.items()}

    for k, pct in shares.items():
        print(f"  {k:<16}: {pct:6.2f}%")
    print()
    return shares