"""
Demo script for Qwen3 implementation.

This script demonstrates how to:
1. Initialize a Qwen3 model
2. Load pretrained weights
3. Set up the tokenizer
4. Generate text
5. Run interactive chat
"""

import torch
from pathlib import Path

from .model import Qwen3Model
from .config import QWEN_CONFIG_06_B
from .tokenizer import Qwen3Tokenizer
from .utils import load_weights_into_qwen, get_device, print_model_info
from .generation import generate_text, interactive_chat


def download_model_weights(use_reasoning_model=True):
    """
    Download Qwen3 model weights and tokenizer.
    
    Args:
        use_reasoning_model: Whether to use reasoning model (True) or base model (False)
        
    Returns:
        Tuple of (model_dir, tokenizer_file)
    """
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError(
            "Required packages not installed. Run: pip install huggingface_hub safetensors"
        )
    
    # Choose repository based on model type
    if use_reasoning_model:
        repo_id = "Qwen/Qwen3-0.6B"
        model_dir = "Qwen3-0.6B"
        tokenizer_file = f"{model_dir}/tokenizer.json"
    else:
        repo_id = "Qwen/Qwen3-0.6B-Base"
        model_dir = "Qwen3-0.6B-Base"
        tokenizer_file = f"{model_dir}/tokenizer.json"
    
    print(f"Downloading weights from {repo_id}...")
    
    # Download model weights
    weights_file = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        local_dir=model_dir
    )
    
    # Download tokenizer
    tokenizer_path = hf_hub_download(
        repo_id=repo_id,
        filename="tokenizer.json",
        local_dir=model_dir
    )
    
    # Load weights
    weights_dict = load_file(weights_file)
    
    print(f"Successfully downloaded model to {model_dir}")
    return weights_dict, tokenizer_file


def setup_model_and_tokenizer(use_reasoning_model=True, device=None):
    """
    Set up Qwen3 model and tokenizer with pretrained weights.
    
    Args:
        use_reasoning_model: Whether to use reasoning model
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = get_device()
    
    print("Setting up Qwen3 model...")
    
    # Initialize model
    model = Qwen3Model(QWEN_CONFIG_06_B)
    print_model_info(model)
    
    # Download and load weights
    weights_dict, tokenizer_file = download_model_weights(use_reasoning_model)
    load_weights_into_qwen(model, QWEN_CONFIG_06_B, weights_dict)
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Set up tokenizer
    if use_reasoning_model:
        repo_id = "Qwen/Qwen3-0.6B"
    else:
        repo_id = "Qwen/Qwen3-0.6B-Base"
        
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file,
        repo_id=repo_id,
        add_generation_prompt=use_reasoning_model,
        add_thinking=use_reasoning_model
    )
    
    # Validate tokenizer
    validation_result = tokenizer.validate_tokenizer()
    if validation_result["status"] == "success":
        print("Tokenizer validation successful!")
    else:
        print(f"Tokenizer validation failed: {validation_result['error']}")
    
    print(f"Model loaded on device: {device}")
    return model, tokenizer


def run_demo():
    """Run a complete demonstration of Qwen3 capabilities."""
    print("=" * 60)
    print("Qwen3 Implementation Demo")
    print("=" * 60)
    
    # Setup
    use_reasoning_model = True
    device = get_device()
    
    try:
        model, tokenizer = setup_model_and_tokenizer(use_reasoning_model, device)
        
        # Demo prompts
        demo_prompts = [
            "What is artificial intelligence?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about programming.",
            "What are the key differences between Python and JavaScript?"
        ]
        
        print("\n" + "=" * 60)
        print("Running demo with sample prompts...")
        print("=" * 60)
        
        for i, prompt in enumerate(demo_prompts, 1):
            print(f"\n--- Demo {i}/{len(demo_prompts)} ---")
            
            result = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.7,
                top_k=50,
                device=device,
                verbose=True
            )
            
            print(f"\nOutput:\n{result['output_text']}")
            print("-" * 40)
        
        # Interactive chat option
        print("\n" + "=" * 60)
        print("Demo completed! Options:")
        print("1. Enter 'chat' for interactive chat session")
        print("2. Enter 'quit' to exit")
        print("=" * 60)
        
        while True:
            choice = input("\nYour choice: ").strip().lower()
            
            if choice == "chat":
                interactive_chat(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_k=50
                )
                break
            elif choice in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            else:
                print("Please enter 'chat' or 'quit'")
                
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install torch huggingface_hub safetensors tokenizers")


if __name__ == "__main__":
    run_demo()