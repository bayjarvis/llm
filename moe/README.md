# Mixture of Experts (MoE) in PyTorch

This directory contains a PyTorch implementation of a sparse Mixture of Experts (MoE) layer.

## Core Concepts

A Mixture of Experts is a neural network architecture that, instead of using a single, large, dense network to process all data, uses multiple smaller, specialized networks called "experts." A "gating network" or "router" determines which experts are best suited to process a given input. This allows the model to learn specialized representations for different types of data, leading to more efficient training and inference. In a **sparse** MoE, only a subset of the experts (the "top-k") are activated for each input, further improving efficiency.

## Implementation Details

The implementation is divided into two main classes: `Expert` and `SparseMoE`.

### 1. The `Expert` Module

- **Purpose:** This is a standard feed-forward neural network that acts as one of the specialized learners in the MoE layer.
- **Architecture:** It's a simple multi-layer perceptron (MLP) with one hidden layer and a ReLU activation function.

### 2. The `SparseMoE` Module

This is the core of the implementation, orchestrating the experts and the gating mechanism.

- **Initialization (`__init__`):**
    - It creates a `ModuleList` of `Expert` networks.
    - It defines the `gating_network`, a single linear layer that outputs a score for each expert.

- **Forward Pass (`forward`):**
    1.  **Gating Scores:** The input is passed through the `gating_network` to get a score for each expert.
    2.  **Top-k Selection:** `torch.topk` selects the indices and scores of the `top_k` experts with the highest scores.
    3.  **Score Normalization:** The scores of the selected experts are normalized using a `softmax` function to create weights.
    4.  **Weighted Output Combination:** The final output is a weighted sum of the outputs from the selected experts.

## Files

- `moe_model.py`: Contains the `Expert` and `SparseMoE` PyTorch modules.
- `train.py`: A script to train the MoE model on synthetic data.

## Usage

To train the model, run:

```bash
python train.py
```
