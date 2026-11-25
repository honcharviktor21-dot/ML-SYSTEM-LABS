import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models import StudentCNN, set_seed, evaluate_accuracy, count_parameters, get_model_disk_size_mb


def distillation_loss(student_logits, teacher_logits, labels, T: float, alpha: float):
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)
    return alpha * hard_loss + (1.0 - alpha) * soft_loss


def run_distillation_study(teacher: nn.Module, train_loader, test_loader, config, device):
    print("--- Task 4: Knowledge Distillation (T=4, alpha=0.5) ---")
    d_cfg = config["distillation"]
    epochs = d_cfg["epochs"]
    lr = d_cfg["lr"]
    T = d_cfg["temperature"]
    alpha = d_cfg["alpha"]
    seed = config["experiment"]["seed"]

    # 1. Plain Student (Trained without Teacher)
    set_seed(seed)
    plain_student = StudentCNN().to(device)
    opt_plain = torch.optim.Adam(plain_student.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        plain_student.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt_plain.zero_grad()
            loss = criterion(plain_student(x), y)
            loss.backward()
            opt_plain.step()

    plain_acc = evaluate_accuracy(plain_student, test_loader, device)
    plain_params = count_parameters(plain_student)
    plain_size = get_model_disk_size_mb(plain_student)

    # 2. Distilled Student (Trained with Frozen Teacher)
    set_seed(seed)
    distilled_student = StudentCNN().to(device)
    opt_distill = torch.optim.Adam(distilled_student.parameters(), lr=lr)
    teacher.eval()

    for _ in range(epochs):
        distilled_student.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt_distill.zero_grad()
            student_logits = distilled_student(x)
            with torch.no_grad():
                teacher_logits = teacher(x)
            loss = distillation_loss(student_logits, teacher_logits, y, T=T, alpha=alpha)
            loss.backward()
            opt_distill.step()

    distilled_acc = evaluate_accuracy(distilled_student, test_loader, device)
    torch.save(distilled_student.state_dict(), "results/student_distilled.pt")

    print(f"  Plain Student     | Params={plain_params:,} | Size={plain_size:.3f} MB | Acc={plain_acc:.4f}")
    print(f"  Distilled Student | Params={plain_params:,} | Size={plain_size:.3f} MB | Acc={distilled_acc:.4f} (+{(distilled_acc - plain_acc)*100:.2f}%)\n")

    return {
        "plain_student": {
            "params": plain_params,
            "size_mb": plain_size,
            "accuracy": plain_acc,
        },
        "distilled_student": {
            "params": plain_params,
            "size_mb": plain_size,
            "accuracy": distilled_acc,
            "accuracy_gain_pct": round((distilled_acc - plain_acc) * 100, 2),
        }
    }, plain_student, distilled_student