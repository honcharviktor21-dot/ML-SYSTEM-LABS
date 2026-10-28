import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ID: 72014 - Lab 2: Neural Network from Scratch

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class NeuralNetwork:
    def __init__(self, hidden_width):
        self.W = hidden_width
        self.params = {
            'W1': np.random.normal(0, 0.01, (64, self.W)),
            'b1': np.zeros((self.W,)),
            'W2': np.random.normal(0, 0.01, (self.W, self.W)),
            'b2': np.zeros((self.W,)),
            'W3': np.random.normal(0, 0.01, (self.W, 10)),
            'b3': np.zeros((10,))
        }

    def forward(self, X):
        self.cache = {'X': X}
        self.cache['Z1'] = np.dot(X, self.params['W1']) + self.params['b1']
        self.cache['A1'] = relu(self.cache['Z1'])
        self.cache['Z2'] = np.dot(self.cache['A1'], self.params['W2']) + self.params['b2']
        self.cache['A2'] = relu(self.cache['Z2'])
        self.cache['Z3'] = np.dot(self.cache['A2'], self.params['W3']) + self.params['b3']
        self.cache['A3'] = softmax(self.cache['Z3'])
        return self.cache['A3']

    def compute_loss(self, Y_pred, Y_true):
        m = Y_true.shape[0]
        Y_pred = np.clip(Y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(Y_true * np.log(Y_pred)) / m

    def backward(self, Y_true):
        m = Y_true.shape[0]
        grads = {}
        dZ3 = self.cache['A3'] - Y_true
        grads['W3'] = np.dot(self.cache['A2'].T, dZ3) / m
        grads['b3'] = np.sum(dZ3, axis=0) / m
        dA2 = np.dot(dZ3, self.params['W3'].T)
        dZ2 = dA2 * relu_deriv(self.cache['Z2'])
        grads['W2'] = np.dot(self.cache['A1'].T, dZ2) / m
        grads['b2'] = np.sum(dZ2, axis=0) / m
        dA1 = np.dot(dZ2, self.params['W2'].T)
        dZ1 = dA1 * relu_deriv(self.cache['Z1'])
        grads['W1'] = np.dot(self.cache['X'].T, dZ1) / m
        grads['b1'] = np.sum(dZ1, axis=0) / m
        return grads

    def update_params(self, grads, lr):
        for key in self.params:
            self.params[key] -= lr * grads[key]

def gradient_check():
    print("--- Running Gradient Check ---")
    X_dummy = np.random.randn(5, 64)
    Y_dummy = np.eye(10)[np.random.choice(10, 5)]
    nn = NeuralNetwork(16)
    nn.forward(X_dummy)
    analytic_grads = nn.backward(Y_dummy)
    
    epsilon = 1e-5
    passed = 0
    for i in range(10):
        layer = np.random.choice(['W1', 'W2', 'W3'])
        shape = nn.params[layer].shape
        idx = tuple(np.random.randint(0, d) for d in shape)
        orig_val = nn.params[layer][idx]
        
        nn.params[layer][idx] = orig_val + epsilon
        loss_plus = nn.compute_loss(nn.forward(X_dummy), Y_dummy)
        
        nn.params[layer][idx] = orig_val - epsilon
        loss_minus = nn.compute_loss(nn.forward(X_dummy), Y_dummy)
        
        nn.params[layer][idx] = orig_val 
        numeric_grad = (loss_plus - loss_minus) / (2 * epsilon)
        analytic_grad = analytic_grads[layer][idx]
        
        rel_error = abs(numeric_grad - analytic_grad) / max(1e-8, abs(numeric_grad) + abs(analytic_grad))
        if rel_error < 1e-6:
            passed += 1
            
    print(f"Gradient check passed for {passed}/10 weights.\n")

def run_experiments():
    os.makedirs('results', exist_ok=True)
    sys.stdout = Logger("results/training_log.txt")
    
    digits = load_digits()
    X = digits.data / 16.0
    y = np.eye(10)[digits.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    
    temp_nn = NeuralNetwork(16)
    preds = temp_nn.forward(X_train)
    initial_loss = temp_nn.compute_loss(preds, y_train)
    print(f"Initial untrained loss: {initial_loss:.4f} (Expected ~2.3026)\n")
    
    widths = [16, 64]
    learning_rates = [0.001, 0.01, 0.1]
    epochs = 50
    batch_size = 32
    
    plt.figure(figsize=(12, 8))
    for W in widths:
        for lr in learning_rates:
            nn = NeuralNetwork(W)
            losses = []
            for epoch in range(epochs):
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]
                
                epoch_loss = 0
                batches = 0
                for i in range(0, X_train.shape[0], batch_size):
                    X_batch = X_shuffled[i:i+batch_size]
                    y_batch = y_shuffled[i:i+batch_size]
                    
                    preds = nn.forward(X_batch)
                    loss = nn.compute_loss(preds, y_batch)
                    grads = nn.backward(y_batch)
                    nn.update_params(grads, lr)
                    epoch_loss += loss
                    batches += 1
                losses.append(epoch_loss / batches)
            
            test_preds = nn.forward(X_test)
            test_acc = np.mean(np.argmax(test_preds, axis=1) == np.argmax(y_test, axis=1))
            print(f"W={W}, LR={lr} | Final Test Accuracy: {test_acc*100:.2f}%")
            plt.plot(range(epochs), losses, label=f'W={W}, LR={lr}')
            
    plt.title('Training Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.savefig('results/loss_curves.png')
    print("\nLoss curves saved to results/loss_curves.png")

    nn_time = NeuralNetwork(64)
    X_sample = X_train[:32]
    Y_sample = y_train[:32]
    fwd_times, bwd_times = [], []
    
    for _ in range(100):
        start = time.perf_counter()
        nn_time.forward(X_sample)
        fwd_times.append(time.perf_counter() - start)
        
        start = time.perf_counter()
        nn_time.backward(Y_sample)
        bwd_times.append(time.perf_counter() - start)
        
    med_fwd = np.median(fwd_times) * 1000
    med_bwd = np.median(bwd_times) * 1000
    
    print(f"\nMedian Forward Pass Time: {med_fwd:.4f} ms")
    print(f"Median Backward Pass Time: {med_bwd:.4f} ms")
    print("Conclusion: The backward pass dominates time complexity because it necessitates additional matrix transpositions and multiplications to compute gradients for both the weights and the upstream activations.")

    # Second Graphic: Timing Bar Chart
    plt.figure(figsize=(8, 6))
    plt.bar(['Forward Pass', 'Backward Pass'], [med_fwd, med_bwd], color=['#1f77b4', '#ff7f0e'])
    plt.ylabel('Median Time (ms)')
    plt.title('Execution Time Comparison: Forward vs. Backward Pass')
    plt.savefig('results/timing_chart.png')
    print("Timing chart saved to results/timing_chart.png\n")

if __name__ == "__main__":
    gradient_check()
    run_experiments()