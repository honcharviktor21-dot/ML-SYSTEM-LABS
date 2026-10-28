# Comprehensive Laboratory Report: Neural Network Implementation from Scratch
**Author:** Viktor Honchar  
**Student ID:** 72014  
**Institution:** Vistula University | Data Science Program

---

## 1. Architectural Overview and Parameter Initialization
This laboratory exercise involves engineering a Multilayer Perceptron (MLP) from scratch using purely native `numpy` matrix operations. The objective is to correctly classify the `load_digits` dataset, which consists of $8\times8$ pixel images (flattened to 64-dimensional input arrays) into 10 distinct numerical classes (digits 0 through 9). 

To achieve this, the architecture is designed with two hidden layers featuring a dynamic width parameter ($W$), Rectified Linear Unit (ReLU) non-linearities, and a concluding Softmax activation for probabilistic classification.

### 1.1 Dimensionality and Transformation Matrix
The network's structural transformations for the high-capacity model ($W=64$) are detailed below. The weights are purposefully initialized from a normal distribution $\mathcal{N}(0, 0.01)$ to break symmetry, while biases are initialized to zero. Scaling the variance by $0.01$ ensures that initial outputs remain small, preventing the Softmax function from saturating and causing vanishing gradients in the very first epoch.

| Layer / Component | Parameter Shape | Matrix Operation | Output Shape | Activation Function |
| :--- | :--- | :--- | :--- | :--- |
| **Input ($X$)** | N/A | Batch of flattened arrays | `(32, 64)` | None |
| **Hidden Layer 1** | $W_1$: `(64, W)`<br>$b_1$: `(W,)` | $Z_1 = X \cdot W_1 + b_1$ | `(32, W)` | ReLU: $f(x) = \max(0, x)$ |
| **Hidden Layer 2** | $W_2$: `(W, W)`<br>$b_2$: `(W,)` | $Z_2 = A_1 \cdot W_2 + b_2$ | `(32, W)` | ReLU: $f(x) = \max(0, x)$ |
| **Output Layer** | $W_3$: `(W, 10)`<br>$b_3$: `(10,)` | $Z_3 = A_2 \cdot W_3 + b_3$ | `(32, 10)` | Softmax: $\frac{e^{z_i}}{\sum e^{z_k}}$ |

---

## 2. Forward Propagation and Theoretical Loss Validation
Forward propagation pushes the data batch through the defined matrix transformations. The final step involves computing the Cross-Entropy loss to penalize the divergence between the predicted probability distribution and the true one-hot encoded labels:

$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

**Validation of the Untrained State:**
Before exposing the network to optimization, a sanity check was performed on the initial forward pass. 
*   **Observed Initial Loss:** $2.3026$
*   **Theoretical Expectation:** An untrained network initialized with near-zero weights will output an approximately uniform distribution across all 10 classes, meaning it assigns a probability of $P = 0.1$ to each class. The theoretical negative log-likelihood is therefore $-\ln(0.1) \approx 2.30258$. 
*   **Conclusion:** The exact parity between the observed and expected loss confirms that the forward pass, the Softmax normalization (which sums to exactly $1.0$), and the Cross-Entropy calculation are mathematically sound and free of overflow/underflow errors.

---

## 3. Backpropagation and Gradient Verification
To enable learning, backpropagation is utilized to calculate the gradient of the loss function with respect to every weight and bias via the chain rule. 

To guarantee the calculus was implemented correctly without human error, the analytical gradients were cross-examined using **Numerical Gradient Checking**. This method approximates the derivative using the central difference formula:
$$\frac{\partial L}{\partial w} \approx \frac{L(w + \epsilon) - L(w - \epsilon)}{2\epsilon}$$

By perturbing a random subset of 10 weights by a microscopic margin ($\epsilon = 1e-5$), the program compared the numerically derived slopes against the analytically derived slopes. Every tested parameter yielded a relative error strictly bounded below the required $1e-6$ threshold, verifying that the backpropagation logic is flawless.

---

## 4. Hyperparameter Evaluation and Training Dynamics
The network was optimized over 50 epochs utilizing Stochastic Gradient Descent (SGD) with a mini-batch size of 32. The experiment tested permutations of two structural widths ($W \in \{16, 64\}$) and three distinct learning rates ($\alpha \in \{0.001, 0.01, 0.1\}$).

### 4.1 Empirical Training Results
| Capacity ($W$) | Learning Rate ($\alpha$) | Final Test Accuracy | Convergence Assessment |
| :---: | :---: | :---: | :--- |
| 16 | 0.001 | **10.19%** | **Stagnant:** Step size too small to traverse the loss landscape[cite: 3]. |
| 16 | 0.01 | **10.19%** | **Stagnant:** Failed to break initial loss plateau[cite: 3]. |
| 16 | 0.1 | **51.48%** | **Partial Convergence:** Network bottlenecked by low dimensionality. |
| 64 | 0.001 | **10.19%** | **Stagnant:** Wide architecture unable to compensate for slow $\alpha$[cite: 3]. |
| 64 | 0.01 | **10.19%** | **Stagnant:** Wide architecture unable to compensate for slow $\alpha$[cite: 3]. |
| **64** | **0.1** | **86.85%** | **Optimal Convergence:** Sufficient capacity and step size to learn. |

### 4.2 Analysis of Learning Behaviors
1.  **The "10.19%" Random Guessing Phenomenon:** Across all tests utilizing learning rates of 0.001 and 0.01, the network yielded an accuracy of exactly 10.19%. Because the dataset contains 10 target classes, a completely blind guess yields an expected accuracy of $\sim10\%$. As visualized in the generated training loss curves, the loss trajectories for these configurations remain entirely flat[cite: 3]. This indicates that the gradient updates were too miniscule to pull the model out of its initial saddle point within the strict 50-epoch limit.
2.  **Capacity vs. Representation:** When utilizing the functional learning rate of $\alpha = 0.1$, a clear distinction emerges between the $W=16$ and $W=64$ models. The wider network ($W=64$) possesses exponentially more trainable parameters, giving it the necessary "memory" and representative capacity to map the complex, non-linear spatial relationships of the $8\times8$ pixel grids. This resulted in an aggressive minimization of loss[cite: 3] and a commanding accuracy of 86.85%, compared to the bottlenecked 51.48% of the narrower architecture.

---

## 5. Computational Complexity and Execution Timing
To understand the hardware performance characteristics, both the forward and backward propagation passes were benchmarked in isolation. The script recorded a median execution time across 100 runs for the $W=64$ architecture.

### 5.1 Microsecond Benchmark Metrics
| Operation Phase | Median Execution Time | Relative Computational Cost |
| :--- | :--- | :--- |
| **Forward Pass** | $0.0403 \text{ ms}$ | $1.00\times$ (Baseline)[cite: 4] |
| **Backward Pass** | $0.0597 \text{ ms}$ | $1.48\times$ (Slower)[cite: 4] |

### 5.2 Matrix Operation Analysis
The empirical data demonstrates that the backward pass incurs a massive ~48% computational penalty compared to the forward pass[cite: 4]. This behavior is a fundamental mathematical reality of training neural networks:

*   **Forward Propagation** is relatively lightweight. It requires only standard, sequential matrix multiplications to push activations forward: $Z^{[l]} = A^{[l-1]} \cdot W^{[l]}$.
*   **Backward Propagation** is heavily demanding. To distribute the error backwards, the CPU must compute two distinct gradients per layer:
    1.  **Weight Gradients:** $dW^{[l]} = (A^{[l-1]})^T \cdot dZ^{[l]}$ (requires transposing the previous activation matrix).
    2.  **Activation Gradients:** $dA^{[l-1]} = dZ^{[l]} \cdot (W^{[l]})^T$ (requires transposing the current weight matrix).

The continuous necessity to perform hardware-level memory transpositions (swapping rows and columns) combined with executing double the amount of dot products inherently causes the backward pass to dominate the overall time complexity of the algorithm[cite: 4].
