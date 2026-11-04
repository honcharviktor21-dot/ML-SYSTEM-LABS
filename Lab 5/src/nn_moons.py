import time
import statistics
import random
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from src.engine import Value

class ScalarNeuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-0.5, 0.5)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.sigmoid()

    def parameters(self):
        return self.w + [self.b]

class ScalarMLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [ [ScalarNeuron(sz[i]) for _ in range(sz[i+1])] for i in range(len(nouts)) ]

    def __call__(self, x):
        for layer in self.layers:
            x = [neuron(x) for neuron in layer]
        return x[0]

    def parameters(self):
        return [p for layer in self.layers for neuron in layer for p in neuron.parameters()]

def run_moons_benchmark(config):
    print("--- Task 5: make_moons Benchmark (2->8->1 MLP) ---")
    seed = config['experiment']['seed']
    random.seed(seed)
    torch.manual_seed(seed)

    X_np, y_np = make_moons(n_samples=config['moons_training']['n_samples'], 
                            noise=config['moons_training']['noise'], 
                            random_state=seed)

    epochs = config['moons_training']['epochs']
    lr = config['moons_training']['lr']
    runs = config['moons_training']['runs']

    # Benchmark Custom Engine
    custom_times = []
    for run in range(runs):
        model_custom = ScalarMLP(2, [8, 1])
        start = time.perf_counter()
        for epoch in range(epochs):
            total_loss = Value(0.0)
            for xi, yi in zip(X_np, y_np):
                x_val = [Value(xi[0]), Value(xi[1])]
                pred = model_custom(x_val)
                # Binary Cross-Entropy Loss term
                loss_i = -(Value(yi) * pred.log() + Value(1 - yi) * (Value(1.0) - pred).log())
                total_loss = total_loss + loss_i
            loss = total_loss / len(X_np)
            for p in model_custom.parameters():
                p.grad = 0.0
            loss.backward()
            for p in model_custom.parameters():
                p.data -= lr * p.grad
        custom_times.append(time.perf_counter() - start)

    # Benchmark PyTorch Equivalent
    X_torch = torch.tensor(X_np, dtype=torch.float32)
    y_torch = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    
    torch_times = []
    for run in range(runs):
        model_torch = nn.Sequential(nn.Linear(2, 8), nn.Sigmoid(), nn.Linear(8, 1), nn.Sigmoid())
        optimizer = torch.optim.SGD(model_torch.parameters(), lr=lr)
        criterion = nn.BCELoss()

        start = time.perf_counter()
        for epoch in range(epochs):
            optimizer.zero_grad()
            preds = model_torch(X_torch)
            loss = criterion(preds, y_torch)
            loss.backward()
            optimizer.step()
        torch_times.append(time.perf_counter() - start)

    custom_median = statistics.median(custom_times)
    torch_median = statistics.median(torch_times)
    speedup = custom_median / torch_median

    print(f"Custom Autograd MLP Median Time ({runs} runs): {custom_median:.4f} s")
    print(f"PyTorch Vectorized MLP Median Time ({runs} runs): {torch_median:.4f} s")
    print(f"PyTorch Vectorization Speedup: {speedup:.2f}x\n")

    return {
        "custom_median_sec": custom_median,
        "torch_median_sec": torch_median,
        "pytorch_speedup_factor": speedup
    }