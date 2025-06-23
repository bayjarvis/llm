"""
Rotary Position Embedding (RoPE) implementation for Qwen3.

RoPE encodes positional information by rotating the query and key vectors
in the attention mechanism, allowing the model to better understand
the relative positions of tokens.
"""

import torch


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    """
    Compute the sine and cosine values for Rotary Position Embedding (RoPE).
    
    Args:
        head_dim: Dimension of each attention head
        theta_base: Base value for computing rotation angles
        context_length: Maximum sequence length
        dtype: Data type for computations
    
    Returns:
        Tuple of (cos, sin) tensors for RoPE
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    """
    Apply Rotary Position Embedding to input tensor.
    
    Args:
        x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
        cos: Cosine values from compute_rope_params
        sin: Sine values from compute_rope_params
    
    Returns:
        Tensor with RoPE applied
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes to match input
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    # The rotation formula: x_rotated = x * cos + (-x2, x1) * sin
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # Preserve the original dtype
    return x_rotated.to(dtype=x.dtype)