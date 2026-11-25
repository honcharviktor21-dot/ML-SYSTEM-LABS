# Individual Project Proposal: PokeEdge-Net — Offline Real-Time Pokémon TCG Card Type, Finish & Binder-Sort Edge Application

**Student Name:** Viktor Honchar  
**Student ID:** 72014  
**Course:** Machine Learning Systems Engineering  
**Target Deployment Runtime:** Local CPU Edge Inference Engine (Laptop Webcam / Mobile Scanner / Retail Sorting Kiosk)  

---

## 1. Problem Statement & Medium-Difficulty Engineering Scope
Collectors, local trading card shops, and tournament players routinely sort thousands of bulk pulls and binder additions by energy type, card category, and foil finish. Existing mobile scanning apps rely on cloud-hosted Vision Transformers or remote OCR lookup APIs, introducing $150\text{–}400\text{ ms}$ network round-trip latency, failing inside concrete-walled game shops with weak Wi-Fi, and incurring recurring cloud inference costs.

This project designs, trains, compresses, and deploys **PokeEdge-Net**, a low-latency edge computer vision application capable of recognizing Pokémon TCG cards in real time directly on CPU hardware without internet connectivity. The problem is deliberately scoped at **medium systems and ML difficulty**:
* **Why it is non-trivial:** Unlike clean studio benchmarks (e.g., Fashion-MNIST or CIFAR-10), ~70% of a Pokémon card's surface is occupied by diverse, non-standardized illustration artwork that varies across every expansion. Full-Art, Illustration Rare, and V/EX cards eliminate standard colored borders entirely, while real-world tabletop captures introduce specular glare off protective polypropylene penny sleeves and rigid toploaders, $\pm 15^\circ$ perspective skew, and subtle color overlaps (*Psychic* vs. *Dragon*, *Darkness* vs. *Metal*).
* **Why it is edge-tractable:** By fusing global card layout cues with a dedicated high-resolution crop of the top-right energy symbol and border semantics at $128 \times 128$ RGB resolution, a compressed dual-branch CNN can achieve $>91\%$ Top-1 accuracy under a strict $<5\text{ MB}$ disk footprint and $<50\text{ ms}$ end-to-end latency budget.

---

## 2. Dataset Architecture, Data Sources & Edge Augmentations

### A. Benchmark Data Sources & Relational Join Pipeline
To ensure full reproducibility and realistic physical-world generalization, the dataset is built by joining three concrete public repositories:
1. **Primary Card Scan Repository (`ellimaaac/pokemon-tcg-all-image-cards` on Kaggle):** 21,340+ high-resolution RGB card images spanning 197 expansions from classic sets through modern releases (*Scarlet & Violet*, *Chaos Rising*, and *30th Celebration*).
2. **Ground-Truth Metadata Table (`priyamchoksi/pokemon-cards` on Kaggle):** Structured tabular dataset providing `supertype` (`Pokémon`, `Trainer`, `Energy`),