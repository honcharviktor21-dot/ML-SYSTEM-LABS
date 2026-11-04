import torch
import json
import os
from src.engine import Value

def target_func(a, b, c):
    return ((a * b + c) ** 2).relu()

def verify_gradients(config):
    print("--- Task 4: Gradient Verification ---")
    a_val = config['engine_test']['test_inputs']['a']
    b_val = config['engine_test']['test_inputs']['b']
    c_val = config['engine_test']['test_inputs']['c']
    h = 1e-6

    # 1. Custom Autograd Engine
    a_custom = Value(a_val)
    b_custom = Value(b_val)
    c_custom = Value(c_val)
    out_custom = target_func(a_custom, b_custom, c_custom)
    out_custom.backward()

    # 2. PyTorch Autograd
    a_torch = torch.tensor(a_val, requires_grad=True, dtype=torch.float64)
    b_torch = torch.tensor(b_val, requires_grad=True, dtype=torch.float64)
    c_torch = torch.tensor(c_val, requires_grad=True, dtype=torch.float64)
    out_torch = target_func(a_torch, b_torch, c_torch)
    out_torch.backward()

    # 3. Numerical Differentiation (Finite Differences)
    f_num = lambda a, b, c: ((a * b + c) ** 2 if (a * b + c) ** 2 > 0 else 0.0)
    num_grad_a = (f_num(a_val + h, b_val, c_val) - f_num(a_val - h, b_val, c_val)) / (2 * h)
    num_grad_b = (f_num(a_val, b_val + h, c_val) - f_num(a_val, b_val - h, c_val)) / (2 * h)
    num_grad_c = (f_num(a_val, b_val, c_val + h) - f_num(a_val, b_val, c_val - h)) / (2 * h)

    diff_torch_a = abs(a_custom.grad - a_torch.grad.item())
    diff_torch_b = abs(b_custom.grad - b_torch.grad.item())
    diff_torch_c = abs(c_custom.grad - c_torch.grad.item())

    diff_num_a = abs(a_custom.grad - num_grad_a)
    diff_num_b = abs(b_custom.grad - num_grad_b)
    diff_num_c = abs(c_custom.grad - num_grad_c)

    max_diff = max(diff_torch_a, diff_torch_b, diff_torch_c, diff_num_a, diff_num_b, diff_num_c)
    print(f"Custom Gradients:    da={a_custom.grad:.6f}, db={b_custom.grad:.6f}, dc={c_custom.grad:.6f}")
    print(f"PyTorch Gradients:   da={a_torch.grad.item():.6f}, db={b_torch.grad.item():.6f}, dc={c_torch.grad.item():.6f}")
    print(f"Numerical Gradients: da={num_grad_a:.6f}, db={num_grad_b:.6f}, dc={num_grad_c:.6f}")
    print(f"Max Absolute Gradient Difference: {max_diff:.8e}")

    assert max_diff < config['engine_test']['tolerance'], f"Verification Failed! Max diff {max_diff} exceeds tolerance."
    print("Verification Passed (< 1e-6).\n")

    results = {
        "custom_grads": {"da": a_custom.grad, "db": b_custom.grad, "dc": c_custom.grad},
        "torch_grads": {"da": a_torch.grad.item(), "db": b_torch.grad.item(), "dc": c_torch.grad.item()},
        "numerical_grads": {"da": num_grad_a, "db": num_grad_b, "dc": num_grad_c},
        "max_absolute_difference": max_diff,
        "status": "PASSED"
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/validation_results.json", "w") as f:
        json.dump(results, f, indent=4)