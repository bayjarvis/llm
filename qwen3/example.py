"""
Simple example script showing how to use the Qwen3 implementation.

This script demonstrates basic usage without requiring pretrained weights.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from qwen3.model import Qwen3Model
from qwen3.config import QWEN_CONFIG_06_B
from qwen3.tokenizer import Qwen3Tokenizer
from qwen3.generation import generate
from qwen3.utils import get_device, print_model_info


def create_toy_model():
    """Create a small Qwen3 model for demonstration (without pretrained weights)."""
    print("Creating toy Qwen3 model...")
    
    # Create a smaller configuration for demo purposes
    toy_config = {
        "vocab_size": 1000,         # Small vocabulary
        "context_length": 128,      # Short context
        "emb_dim": 256,            # Small embedding dimension
        "n_heads": 8,              # Few attention heads
        "n_layers": 4,             # Few layers
        "hidden_dim": 512,         # Small FFN dimension
        "head_dim": 32,            # Small head dimension
        "qk_norm": True,           # Enable QK normalization
        "n_kv_groups": 4,          # Half the heads for GQA
        "rope_base": 10000.0,      # Standard RoPE base
        "dtype": torch.float32,    # Use float32 for compatibility
    }
    
    model = Qwen3Model(toy_config)
    print_model_info(model)
    
    return model, toy_config


def demonstrate_forward_pass(model, config):
    """Demonstrate a forward pass through the model."""
    print("\n" + "="*40)
    print("Demonstrating Forward Pass")
    print("="*40)
    
    # Create some random input tokens
    batch_size = 2
    seq_length = 10
    vocab_size = config["vocab_size"]
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    print(f"Input shape: {input_ids.shape}")
    print(f"Input tokens: {input_ids[0].tolist()}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_length}, {vocab_size})")
    
    # Get probabilities for the last token
    last_token_logits = logits[0, -1, :]  # First batch, last token
    probs = torch.softmax(last_token_logits, dim=-1)
    top_5_probs, top_5_indices = torch.topk(probs, 5)
    
    print(f"Top 5 predicted tokens for last position:")
    for i, (prob, idx) in enumerate(zip(top_5_probs, top_5_indices)):
        print(f"  {i+1}. Token {idx.item()}: {prob.item():.4f}")


def demonstrate_generation(model, config):
    """Demonstrate text generation."""
    print("\n" + "="*40)
    print("Demonstrating Text Generation")
    print("="*40)
    
    # Start with a random token sequence
    start_tokens = torch.randint(0, config["vocab_size"], (1, 5))
    print(f"Starting tokens: {start_tokens[0].tolist()}")
    
    # Generate text using different strategies
    strategies = [
        {"name": "Greedy", "temperature": 0.0, "top_k": None},
        {"name": "Random", "temperature": 1.0, "top_k": None},
        {"name": "Top-K", "temperature": 0.8, "top_k": 10},
    ]
    
    for strategy in strategies:
        print(f"\n{strategy['name']} Generation:")
        
        generated = generate(
            model=model,
            idx=start_tokens.clone(),
            max_new_tokens=10,
            context_size=config["context_length"],
            temperature=strategy["temperature"],
            top_k=strategy["top_k"]
        )
        
        new_tokens = generated[0, len(start_tokens[0]):].tolist()
        print(f"  Generated tokens: {new_tokens}")
        print(f"  Full sequence: {generated[0].tolist()}")


def demonstrate_chat_formatting():
    """Demonstrate chat message formatting."""
    print("\n" + "="*40)
    print("Demonstrating Chat Formatting")
    print("="*40)
    
    # Test chat formatting without actual tokenizer
    messages = [
        {"role": "user", "content": "What is artificial intelligence?"},
    ]
    
    print("Original messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    
    # Test different formatting options
    print("\nFormatted for base model:")
    formatted_base = Qwen3Tokenizer.format_qwen_chat(
        messages, 
        add_generation_prompt=False, 
        add_thinking=False
    )
    print(repr(formatted_base))
    
    print("\nFormatted for reasoning model:")
    formatted_reasoning = Qwen3Tokenizer.format_qwen_chat(
        messages, 
        add_generation_prompt=True, 
        add_thinking=True
    )
    print(repr(formatted_reasoning))
    
    print("\nFormatted for base model with generation prompt:")
    formatted_base_prompt = Qwen3Tokenizer.format_qwen_chat(
        messages, 
        add_generation_prompt=True, 
        add_thinking=False
    )
    print(repr(formatted_base_prompt))


def demonstrate_model_analysis(model, config):
    """Demonstrate model analysis capabilities."""
    print("\n" + "="*40)
    print("Model Analysis")
    print("="*40)
    
    # Parameter counting
    total_params = sum(p.numel() for p in model.parameters())
    unique_params = model.get_num_params(non_embedding=True)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Unique parameters: {unique_params:,}")
    print(f"Parameter reduction from weight tying: {total_params - unique_params:,}")
    
    # Memory estimation
    memory_info = model.estimate_memory_usage()
    print(f"\nMemory usage ({memory_info['dtype']}):")
    print(f"  Parameters: {memory_info['parameters_gb']:.3f} GB")
    print(f"  Gradients: {memory_info['gradients_gb']:.3f} GB")
    print(f"  Buffers: {memory_info['buffers_gb']:.3f} GB")
    print(f"  Total: {memory_info['total_gb']:.3f} GB")
    
    # Model architecture
    print(f"\nArchitecture details:")
    print(f"  Vocabulary size: {config['vocab_size']:,}")
    print(f"  Context length: {config['context_length']:,}")
    print(f"  Embedding dimension: {config['emb_dim']}")
    print(f"  Number of layers: {config['n_layers']}")
    print(f"  Attention heads: {config['n_heads']}")
    print(f"  KV groups: {config['n_kv_groups']}")
    print(f"  Group size: {config['n_heads'] // config['n_kv_groups']}")


def main():
    """Main demonstration function."""
    print("Qwen3 Implementation Example")
    print("="*50)
    print("This example demonstrates the Qwen3 implementation")
    print("using a small toy model (no pretrained weights).")
    print("="*50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    try:
        # Create toy model
        model, config = create_toy_model()
        
        # Run demonstrations
        demonstrate_forward_pass(model, config)
        demonstrate_generation(model, config)
        demonstrate_chat_formatting()
        demonstrate_model_analysis(model, config)
        
        print("\n" + "="*50)
        print("Example completed successfully!")
        print("\nTo use with pretrained weights:")
        print("1. Install: pip install huggingface_hub safetensors tokenizers")
        print("2. Run: python demo.py")
        print("="*50)
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()