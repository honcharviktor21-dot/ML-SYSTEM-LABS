
## Project Overview
This repository contains the implementation of a two-hidden-layer neural network built entirely from scratch using `numpy`. It is designed to classify the `load_digits` dataset. The project demonstrates the manual implementation of forward propagation, cross-entropy loss, backpropagation (via the chain rule), and analytical gradient validation using numerical gradient checking.

## Repository Structure
* `network.py`: The main Python script containing the `NeuralNetwork` class, training loop, and timing experiments.
* `report.md`: A written summary of the network architecture, gradient checking validation, and computational timing analysis.
* `results/`: Directory containing the execution outputs.
  * `training_log.txt`: Console output detailing initial loss, gradient check status, and final test accuracies across different hyperparameters.
  * `loss_curves.png`: A plot comparing cross-entropy loss over 50 epochs for different learning rates (0.001, 0.01, 0.1) and hidden widths (16, 64).
  * `timing_chart.png`: A bar chart illustrating the computational cost difference between forward and backward passes.

## How to Run
Ensure you have the required libraries installed:
```bash
pip install numpy matplotlib scikit-learn