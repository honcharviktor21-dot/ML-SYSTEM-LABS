import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TeacherCNN(nn.Module):
    """Oversized baseline CNN (~1.08M parameters) for Fashion-MNIST (1x28x28)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 160),
            nn.ReLU(),
            nn.Linear(160, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class StudentCNN(nn.Module):
    """Compact student CNN (~106k parameters) for Fashion-MNIST (1x28x28)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CompactedTeacherCNN(nn.Module):
    """Physically slimmed CNN after structured channel removal (for true hardware speedup)."""
    def __init__(self, c1=32, c2=64, fc=80, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, fc),
            nn.ReLU(),
            nn.Linear(fc, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_model_disk_size_mb(model: nn.Module, temp_path: str = "results/_temp_model.pt") -> float:
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 ** 2)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return round(size_mb, 3)


def get_loaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    data_dir = config["data"]["data_dir"]
    train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    subset_size = config["data"].get("subset_size")
    if subset_size is not None and subset_size < len(train_ds):
        train_ds = Subset(train_ds, list(range(subset_size)))
        test_ds = Subset(test_ds, list(range(min(4000, len(test_ds)))))

    bs = config["data"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)
    return train_loader, test_loader


def evaluate_accuracy(model: nn.Module, test_loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return round(correct / total, 4)