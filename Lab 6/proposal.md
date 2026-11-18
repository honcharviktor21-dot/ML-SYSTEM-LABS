# Individual Project Proposal: Edge-Optimized Real-Time Visual Apparel & Defect Classifier

**Student Name:** Viktor Honchar  
**Student ID:** 72014  
**Course:** Machine Learning Systems Engineering  
**Target Deployment Environment:** Resource-Constrained CPU Edge Inference Runtime  

---

## 1. Problem Statement & System Motivation
Automated visual apparel sorting and retail inventory inspection on edge devices—such as warehouse conveyor scanners, smart checkout kiosks, and mobile point-of-sale terminals—require deterministic, real-time image classification without relying on continuous cloud connectivity. Offloading high-frequency camera frames to remote GPU servers introduces unacceptable network round-trip latency ($>100\text{ ms}$), bandwidth bottlenecks, privacy exposure, and recurring cloud infrastructure costs. The objective of this project is to design, train, and compress a lightweight Convolutional Neural Network (`EdgeApparelCNN`) capable of executing accurate, low-latency on-device inference strictly within edge CPU compute and storage budgets.

---

## 2. Dataset & Reproducible Data Pipeline
* **Dataset:** **Fashion-MNIST**, consisting of 70,000 grayscale images ($28 \times 28$ single-channel resolution) balanced across 10 distinct apparel categories (`T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, and `Ankle boot`), partitioned into 60,000 training and 10,000 test samples.
* **Preprocessing & Augmentation:** Reproducible stratified train/validation splitting (`seed = 72014`), channel-wise zero-mean unit-variance normalization ($\mu = 0.2860, \sigma = 0.3530$), and lightweight online training augmentations (random horizontal flips and minor affine translations) to improve robustness against camera angle and lighting variations on physical sorting lines.

---

## 3. Success Metrics & Deployment Constraints (SLAs)
To qualify for production edge deployment, the final compressed model artifact must satisfy both predictive quality targets and strict systems-level Service Level Agreements (SLAs):

| Evaluation Dimension | Target Constraint / SLA | Measurement Protocol |
| :--- | :--- | :--- |
| **Predictive Success Metric** | **Top-1 Accuracy $\ge 90.0\%$** (Macro F1 $\ge 0.895$) | Evaluated across the full 10,000-image held-out test set |
| **Inference Latency** | **$\le 50.0\text{ ms}$** per forward pass | Median & p95 single-image (`batch_size = 1`) CPU latency over 500 runs |
| **Model Footprint** | **$\le 5.0\text{ MB}$** on disk | Serialized quantized model file size (`torch.jit.save` / `state_dict`) |
| **Peak Runtime Memory** | **$\le 25.0\text{ MB}$** RAM | Peak resident activation & weight memory during inference |

---

## 4. Planned Optimization & Compression Strategy
Building on our YAML-driven, MLflow-tracked PyTorch training pipeline (`Adam` optimizer with `CosineAnnealingLR` scheduling), model compression will proceed in three sequential stages:

1. **Compact Baseline Architecture (`FP32`):** Train a streamlined 3-block CNN utilizing $3 \times 3$ convolutions, batch normalization, and global average pooling ($\approx 210\text{K}$ parameters; $\approx 0.84\text{ MB}$ baseline FP32 size) to establish the accuracy ceiling and latency baseline.
2. **Structured Channel Pruning:** Apply $L_1$-norm structured channel pruning (25%–35% filter sparsity across convolutional layers) to permanently remove redundant feature channels and reduce multiply-accumulate operations (MACs), followed by 5 epochs of fine-tuning to recover baseline predictive accuracy.
3. **Post-Training Static Quantization (`INT8`):** Calibrate activation dynamic ranges using a 1,000-sample validation subset and convert weights and activations from 32-bit floating point (`FP32`) to 8-bit integer (`INT8`) precision via `torch.ao.quantization`. This targets an approximate $3.5\times$–$4\times$ disk footprint reduction (well under the $5\text{ MB}$ limit) and accelerated CPU SIMD execution (well under the $50\text{ ms}$ latency ceiling).

---

## 5. Verification & Deliverable Plan
All experimental runs, hyperparameters, and model checkpoints will be logged in MLflow. The final deliverable will present a Pareto trade-off comparison across the **FP32 Baseline**, **Pruned FP32**, and **Pruned + INT8 Quantized** models covering Top-1 accuracy, model size (MB), MAC reduction, and single-threaded CPU latency.