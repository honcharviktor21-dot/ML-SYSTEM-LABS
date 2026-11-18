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


class Lab3CNN(nn.Module):
    """Best lightweight CNN from Lab 3 (1x28x28 -> 16 -> 32 -> 10)."""
    def __init__(self, in_channels=1, c1=16, c2=32, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(config, device):
    m_cfg = config["model"]
    model = Lab3CNN(
        in_channels=m_cfg["in_channels"],
        c1=m_cfg["conv1_channels"],
        c2=m_cfg["conv2_channels"],
        num_classes=m_cfg["num_classes"],
    )
    return model.to(device)


def get_datasets(config):
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
        test_ds = Subset(test_ds, list(range(min(2000, len(test_ds)))))
    return train_ds, test_ds


def get_loaders(config, batch_size, num_workers=0):
    train_ds, test_ds = get_datasets(config)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, test_loader


def evaluate_accuracy(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total