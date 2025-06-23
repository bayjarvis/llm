"""
Utility functions for Qwen3 implementation.
"""

import os
import urllib.request
from pathlib import Path
import torch


def download_from_huggingface(repo_id, filename, local_dir, revision="main"):
    """
    Download a file from Hugging Face Hub.
    
    Args:
        repo_id: Repository ID (e.g., "Qwen/Qwen3-0.6B")
        filename: Name of file to download
        local_dir: Local directory to save the file
        revision: Git revision (branch/tag/commit)
    
    Returns:
        Path to downloaded file
    """
    base_url = "https://huggingface.co"
    url = f"{base_url}/{repo_id}/resolve/{revision}/{filename}"
    
    # Create directory if it doesn't exist
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    dest_path = os.path.join(local_dir, filename)
    
    print(f"Downloading {url} to {dest_path}...")
    
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Successfully downloaded {filename}")
        return dest_path
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        raise


def load_weights_into_qwen(model, param_config, params):
    """
    Load pretrained weights into a Qwen3 model.
    
    Args:
        model: Qwen3Model instance
        param_config: Model configuration dictionary
        params: Dictionary of pretrained weights
    """
    def assign(left, right, tensor_name="unknown"):
        """Helper function to assign weights with shape checking."""
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(right.clone().detach() if isinstance(right, torch.Tensor) else torch.tensor(right))

    # Load token embeddings
    model.tok_emb.weight = assign(
        model.tok_emb.weight, 
        params["model.embed_tokens.weight"], 
        "model.embed_tokens.weight"
    )

    # Load weights for each transformer layer
    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]
        att = block.att

        # Query, Key, Value projections
        att.W_query.weight = assign(
            att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        att.W_key.weight = assign(
            att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        att.W_value.weight = assign(
            att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )

        # Attention output projection
        att.out_proj.weight = assign(
            att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )

        # Query and Key normalization (if enabled)
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.scale = assign(
                att.q_norm.scale,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight"
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.scale = assign(
                att.k_norm.scale,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight"
            )

        # Attention layer normalization
        block.norm1.scale = assign(
            block.norm1.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Feedforward network weights
        block.ff.fc1.weight = assign(
            block.ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        block.ff.fc2.weight = assign(
            block.ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        block.ff.fc3.weight = assign(
            block.ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        
        # Feedforward layer normalization
        block.norm2.scale = assign(
            block.norm2.scale,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Final layer normalization
    model.final_norm.scale = assign(
        model.final_norm.scale, 
        params["model.norm.weight"], 
        "model.norm.weight"
    )

    # Output head (uses weight tying with input embeddings)
    model.out_head.weight = assign(
        model.out_head.weight, 
        params["model.embed_tokens.weight"], 
        "model.embed_tokens.weight (tied)"
    )

    print("Successfully loaded all pretrained weights!")


def get_device():
    """
    Get the best available device for PyTorch operations.
    
    Returns:
        torch.device: The best available device (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def print_model_info(model):
    """
    Print comprehensive information about a Qwen3 model.
    
    Args:
        model: Qwen3Model instance
    """
    total_params = sum(p.numel() for p in model.parameters())
    unique_params = model.get_num_params(non_embedding=True)
    
    print(f"Model Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Unique parameters: {unique_params:,}")
    print(f"  Model config: {model.cfg}")
    
    # Memory usage estimation
    memory_info = model.estimate_memory_usage()
    print(f"  Estimated memory usage ({memory_info['dtype']}):")
    print(f"    Parameters: {memory_info['parameters_gb']:.2f} GB")
    print(f"    Gradients: {memory_info['gradients_gb']:.2f} GB") 
    print(f"    Total: {memory_info['total_gb']:.2f} GB")


def validate_config(config):
    """
    Validate a Qwen3 model configuration.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = [
        "vocab_size", "context_length", "emb_dim", "n_heads", 
        "n_layers", "hidden_dim", "n_kv_groups", "rope_base", "dtype"
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate dimensions
    if config["emb_dim"] % config["n_heads"] != 0:
        raise ValueError("emb_dim must be divisible by n_heads")
        
    if config["n_heads"] % config["n_kv_groups"] != 0:
        raise ValueError("n_heads must be divisible by n_kv_groups")
        
    if config["head_dim"] is not None:
        if config["head_dim"] % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
    
    print("Configuration validation passed!")