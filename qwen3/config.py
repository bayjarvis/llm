"""
Configuration settings for Qwen3 models.
"""

import torch

# Qwen3 0.6B model configuration
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,           # Vocabulary size
    "context_length": 40_960,        # Context length that was used to train the model
    "emb_dim": 1024,                 # Embedding dimension
    "n_heads": 16,                   # Number of attention heads
    "n_layers": 28,                  # Number of layers
    "hidden_dim": 3072,              # Size of the intermediate dimension in FeedForward
    "head_dim": 128,                 # Size of the heads in GQA
    "qk_norm": True,                 # Whether to normalize queries and values in GQA
    "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
    "rope_base": 1_000_000.0,        # The base in RoPE's "theta"
    "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
}

# Alternative configurations for different model sizes
QWEN_CONFIGS = {
    "0.6B": QWEN_CONFIG_06_B,
    # Add more configurations as needed
}

def get_qwen_config(model_size="0.6B"):
    """Get configuration for specified Qwen3 model size."""
    if model_size not in QWEN_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(QWEN_CONFIGS.keys())}")
    return QWEN_CONFIGS[model_size].copy()