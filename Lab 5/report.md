# Laboratory 5: AI Framework Architecture, Automatic Differentiation, and Execution Performance Report

**Student:** Viktor Honchar  
**Student ID:** 72014  
**Course:** Machine Learning Systems Engineering  
**Prescribed Text Reference:** *Introduction to Machine Learning Systems: Principles and Practices of Engineering Artificially Intelligent Systems* by Vijay Janapa Reddi (Chapter 7, §§7.3–7.8)

---

## 1. Executive Summary & Graph Theory Fundamentals

This laboratory report presents an end-to-end engineering evaluation of automatic differentiation (Autograd), dynamic computation graph construction, and framework-level execution optimizations. Machine learning frameworks evaluate complex neural network functions by framing computation as a Directed Acyclic Graph (DAG) $G = (V, E)$, where:
* **Vertices ($V$):** Represent scalar or tensor data structures (variables/intermediates).
* **Edges ($E$):** Represent functional operations that transform parent inputs into child outputs.

To evaluate gradients for backpropagation, reverse-mode automatic differentiation performs a two-pass traversal:
1. **Forward Pass:** Evaluates intermediate values, dynamically constructs parent-child node pointers, and caches values required for local derivative evaluations.
2. **Backward Pass:** Topologically orders the DAG and applies the chain rule in reverse topological sequence from the final scalar loss $L$. 

The multivariate chain rule dictates that for a node $x_i$ influencing multiple downstream children $\{y_1, y_2, \dots, y_k\}$, its total gradient is the sum of partial derivatives accumulated across all outgoing execution paths:
$$\frac{\partial L}{\partial x_i} = \sum_{j=1}^{k} \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}$$

This report documents the implementation of a scalar-level Autograd engine (`Value`), verifies its numerical and analytical correctness against PyTorch's `autograd`, profiles scalar versus C++ vectorized tensor execution on the `make_moons` dataset, and evaluates execution modes (Eager vs. Traced) alongside hardware memory contiguity costs.

---

## 2. Scalar Autograd Engine Implementation Details

### 2.1 Node Data Structure (`Value`)
The custom scalar autograd engine encapsulates scalar float values, gradient stores, dynamic computation dependencies, and local derivative closures within a single `Value` node class:

* **`data`:** Floating-point value computed during the forward pass.
* **`grad`:** Accumulated partial derivative $\frac{\partial L}{\partial x}$, initialized to $0.0$.
* **`_prev`:** A set containing references to parent `Value` nodes involved in generating the current node.
* **`_backward`:** A parameterless lambda function capturing the local chain-rule derivative step.

### 2.2 Local Derivative Rules
For each supported operation, the engine defines the local chain-rule update passed to parent nodes during graph traversal:

1. **Addition ($z = x + y$):**
   $$\frac{\partial L}{\partial x} \mathrel{+}= 1.0 \cdot \frac{\partial L}{\partial z}, \quad \frac{\partial L}{\partial y} \mathrel{+}= 1.0 \cdot \frac{\partial L}{\partial z}$$

2. **Multiplication ($z = x \cdot y$):**
   $$\frac{\partial L}{\partial x} \mathrel{+}= y \cdot \frac{\partial L}{\partial z}, \quad \frac{\partial L}{\partial y} \mathrel{+}= x \cdot \frac{\partial L}{\partial z}$$

3. **Power ($z = x^k$):**
   $$\frac{\partial L}{\partial x} \mathrel{+}= (k \cdot x^{k-1}) \cdot \frac{\partial L}{\partial z}$$

4. **Rectified Linear Unit ($z = \text{ReLU}(x)$):**
   $$\frac{\partial L}{\partial x} \mathrel{+}= \mathbb{I}(x > 0) \cdot \frac{\partial L}{\partial z}, \quad \text{where } \mathbb{I}(x > 0) = \begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}$$

5. **Sigmoid ($\sigma(x) = \frac{1}{1 + e^{-x}}$):**
   $$\frac{\partial L}{\partial x} \mathrel{+}= (\sigma(x) \cdot (1 - \sigma(x))) \cdot \frac{\partial L}{\partial z}$$

6. **Natural Logarithm ($z = \ln(x)$):**
   $$\frac{\partial L}{\partial x} \mathrel{+}= \left(\frac{1}{x}\right) \cdot \frac{\partial L}{\partial z}$$

### 2.3 Reverse Topological Backpropagation
To guarantee that a node processes all incoming gradients from child nodes before updating its parents, the `.backward()` method uses Depth-First Search (DFS) to build a linear topological ordering of the graph. The root node gradient is set to $1.0$ ($\frac{\partial L}{\partial L} = 1.0$), and the list is evaluated in reverse order:

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
        node._backward()
```

---

## 3. Mathematical & Empirical Gradient Verification (Task 4)

To validate the autograd engine against baseline standards, the target function $f(a, b, c)$ was evaluated at test points $a = 2.0$, $b = -3.0$, and $c = 10.0$:
$$f(a, b, c) = \text{ReLU}\left((a \cdot b + c)^2\right)$$

### 3.1 Analytical Derivation
Let $u = a \cdot b = 2.0 \cdot (-3.0) = -6.0$.  
Let $v = u + c = -6.0 + 10.0 = 4.0$.  
Let $w = v^2 = 4.0^2 = 16.0$.  
Since $w = 16.0 > 0$, $\text{ReLU}(w) = 16.0$, meaning the activation unit is active ($\frac{\partial f}{\partial w} = 1.0$).

Applying the analytical chain rule:
$$\frac{\partial f}{\partial c} = \frac{\partial f}{\partial w} \cdot \frac{\partial w}{\partial v} \cdot \frac{\partial v}{\partial c} = 1.0 \cdot (2v) \cdot 1.0 = 2(4.0) = 8.0$$

$$\frac{\partial f}{\partial a} = \frac{\partial f}{\partial w} \cdot \frac{\partial w}{\partial v} \cdot \frac{\partial v}{\partial u} \cdot \frac{\partial u}{\partial a} = 1.0 \cdot (2v) \cdot 1.0 \cdot b = 8.0 \cdot (-3.0) = -24.0$$

$$\frac{\partial f}{\partial b} = \frac{\partial f}{\partial w} \cdot \frac{\partial w}{\partial v} \cdot \frac{\partial v}{\partial u} \cdot \frac{\partial u}{\partial b} = 1.0 \cdot (2v) \cdot 1.0 \cdot a = 8.0 \cdot 2.0 = 16.0$$

### 3.2 Finite Differences Numerical Estimator
Central finite difference approximation was calculated using perturbation step $h = 1.0 \times 10^{-6}$:
$$\frac{\partial f}{\partial x_i} \approx \frac{f(x_i + h) - f(x_i - h)}{2h}$$

### 3.3 Comparative Results Matrix

| Variable | Custom Autograd | PyTorch `autograd` | Numerical Estimate ($h=10^{-6}$) | Absolute Error (vs. PyTorch) |
| :--- | :--- | :--- | :--- | :--- |
| $\frac{\partial f}{\partial a}$ | **-24.000000** | **-24.000000** | -23.99999999891378 | $0.000000 \times 10^{0}$ |
| $\frac{\partial f}{\partial b}$ | **16.000000** | **16.000000** | 16.000000002236447 | $0.000000 \times 10^{0}$ |
| $\frac{\partial f}{\partial c}$ | **8.000000** | **8.000000** | 7.999999993124618 | $0.000000 \times 10^{0}$ |

* **Max Absolute Difference Across All Estimates:** $6.87538204 \times 10^{-9}$
* **Validation Threshold:** $< 1.0 \times 10^{-6}$
* **Status:** **PASSED**

---

## 4. Performance Profiling: Scalar Engine vs. PyTorch Backend (Task 5)

A multi-layer perceptron with architecture $2 \rightarrow 8 \rightarrow 1$ (Sigmoid hidden activation, Sigmoid output activation, Binary Cross-Entropy loss) was trained on the non-linear `make_moons` dataset ($N=100$, noise $\sigma = 0.1$) across 30 training epochs.

### 4.1 Comparative Benchmark Results

```text
Custom Autograd MLP Median Time (5 runs):  0.9814 s
PyTorch Vectorized MLP Median Time (5 runs): 0.0064 s
Vectorized PyTorch Speedup Factor:         153.59x
```

### 4.2 Architectural Bottleneck Analysis
The empirical **153.59x speedup** achieved by PyTorch stems from three architectural factors defined in Reddi Chapter 7:

1. **Python Interpreter & Allocation Overhead:** The custom engine allocates discrete Python `Value` heap objects for every intermediate scalar operation. In contrast, PyTorch delegates memory allocation to a single contiguous buffer managed by C++ backend allocators.
2. **Graph Traversal Complexity:** For $N=100$ samples over 30 epochs, the scalar engine constructs and topologically sorts thousands of scalar nodes per step. PyTorch constructs a single coarse-grained tensor DAG containing only two matrix multiplication nodes per layer.
3. **SIMD Vectorization & BLAS Execution:** PyTorch executes tensor multiplication using highly optimized C++ Basic Linear Algebra Subprograms (BLAS) with AVX-512 SIMD vectorization, executing matrix operations across CPU vector lanes in parallel.

---

## 5. Framework Execution Modes & Memory Layout Analysis (Task 6)

### 5.1 Eager vs. Traced Execution (`torch.jit.trace`)
Evaluated using a 2-layer Linear model ($2000 \times 2000$ matrix size) over 20 iterations following 5 warm-up cycles:

| Execution Mode | Median Iteration Latency | Latency Reduction |
| :--- | :--- | :--- |
| **Eager Execution** | 95.033 ms | Baseline |
| **Traced Execution (`torch.jit.trace`)** | 88.073 ms | **7.32% Reduction** |

*Analysis:* Eager mode incurs overhead from line-by-line Python interpreter dispatch and intermediate C++ bridge validation. TorchScript tracing captures the computation sub-graph into an intermediate representation (IR), allowing the execution runtime to bypass Python overhead and apply kernel optimizations.

### 5.2 Tensor Memory Layout & Contiguity Impact
Evaluated by computing memory reductions on a $3000 \times 3000$ double-precision matrix:

```text
Contiguous Tensor Shape: torch.Size([3000, 3000]), Stride: (3000, 1), Contiguous: True
Transposed Tensor Shape: torch.Size([3000, 3000]), Stride: (1, 3000), Contiguous: False

Contiguous Sum Median Time:     0.779 ms
Non-Contiguous Sum Median Time: 0.805 ms
Contiguity Overhead Ratio:     1.0336x (3.36% Slowdown)
```

*Hardware Memory Dynamics:*
* **Contiguous Array (`stride=(3000, 1)`):** Values along rows occupy adjacent physical RAM addresses. When a CPU core requests a byte, the hardware prefetcher loads an entire 64-byte L1 cache line containing adjacent elements, maximizing memory throughput.
* **Transposed View (`stride=(1, 3000)`):** A matrix transpose alters tensor metadata (strides) without reordering underlying physical RAM storage. Successive element access requires jumping 3000 float positions, triggering frequent L1/L2 cache misses and forcing RAM fetch penalties.

---

## 6. Engineering Synthesis: Guidelines for Framework Compilation

Graph compilation (`torch.compile`, TorchScript tracing, or XLA compilation) provides distinct benefits depending on model workload:

### 6.1 When Compilation Yields Significant Speedups
1. **Element-Wise Operator Chains:** Sequences of element-wise operations (e.g., $\text{ReLU}(\text{Add}(\text{Mul}(X, W), B))$) are fused into a single fused execution kernel. This avoids writing intermediate tensors back to main memory, keeping operations in high-speed GPU SRAM or CPU registers.
2. **Static Tensor Shapes:** When shapes are invariant, compilation optimizes loop unrolling, register allocation, and static memory scheduling without dynamic allocation overhead.
3. **Small Batch Inference & Light Compute Workloads:** When execution latency is dominated by Python overhead rather than matrix compute limits, eliminating dispatcher latency yields major speed improvements.

### 6.2 When Compilation Provides Minimal Benefit
1. **Compute-Bound Operations (Large GEMM / MatMul):** When execution runtime is strictly bottlenecked by raw floating-point operations in cuBLAS or OpenBLAS matrix kernels, compiler fusion provides minimal additional gain.
2. **Highly Dynamic Graphs:** Dynamic sequence lengths or conditional branching trigger frequent re-compilations (compilation cache misses), introducing overhead that can exceed eager execution time.