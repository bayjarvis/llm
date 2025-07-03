# src/mlx_grpo/chat.py

import argparse
import sys
import os
import json
from typing import List, Dict

import mlx.core as mx
from mlx_lm import load, generate

def load_custom_adapter(model, adapter_path):
    """
    Custom adapter loading function that applies GRPO adapter weights to the model.
    """
    if not os.path.exists(adapter_path):
        return False
    
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    config_file = os.path.join(adapter_path, "adapter_config.json")
    
    if not os.path.exists(adapter_file) or not os.path.exists(config_file):
        print(f"Missing adapter files in {adapter_path}")
        return False
    
    # Load adapter weights
    try:
        adapter_weights = mx.load(adapter_file)
        print(f"Loaded {len(adapter_weights)} adapter parameters")
        
        # Apply adapter weights to model
        def apply_adapter_to_params(model_params, adapter_weights, prefix=""):
            for key, value in model_params.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    apply_adapter_to_params(value, adapter_weights, full_key)
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            apply_adapter_to_params(item, adapter_weights, f"{full_key}.{i}")
                elif hasattr(value, 'dtype') and full_key in adapter_weights:
                    # Apply adapter delta to original parameter
                    print(f"Applying adapter to {full_key}")
                    model_params[key] = value + adapter_weights[full_key]
        
        # Apply to model parameters
        model_params = model.parameters()
        apply_adapter_to_params(model_params, adapter_weights)
        
        return True
        
    except Exception as e:
        print(f"Error applying adapter: {e}")
        return False

def main():
    """
    Main function to run an interactive chat session with a fine-tuned MLX model.
    """
    parser = argparse.ArgumentParser(
        description="Chat with a fine-tuned MLX model."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="The path to the base model's directory or its Hugging Face repo ID.",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="grpo_adapter",
        help="Path to the trained adapter directory from GRPO training (e.g., grpo_adapter).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=250,
        help="The maximum number of tokens to generate in a single response.",
    )
    parser.add_argument(
        "--temp", 
        type=float, 
        default=0.7, 
        help="The sampling temperature for generation."
    )
    args = parser.parse_args()

    # Load the base model and tokenizer
    try:
        print("Loading base model...")
        model, tokenizer = load(args.model)
        
        # Apply custom adapter if specified
        if args.adapter_file:
            print(f"Loading custom adapter: {args.adapter_file}")
            try:
                adapter_loaded = load_custom_adapter(model, args.adapter_file)
                if adapter_loaded:
                    print("✅ Custom adapter loaded successfully!")
                else:
                    print("⚠️  Adapter file not found, using base model")
            except Exception as adapter_error:
                print(f"⚠️  Adapter loading failed: {adapter_error}")
                print("Using base model without adapter...")
        else:
            print("Using base model (no adapter specified)")
            
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        sys.exit(1)

    print("\nModel and adapter loaded successfully.")
    print("Start chatting with your fine-tuned model (type 'exit' or 'quit' to end).\n")

    # This list will store the conversation history.
    history: List = []

    while True:
        try:
            user_prompt = input("You: ")
            if user_prompt.lower() in ["exit", "quit"]:
                break
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C or Ctrl+D to exit gracefully.
            break

        history.append({"role": "user", "content": user_prompt})

        # The tokenizer's chat template formats the entire conversation history
        # so the model understands the context of the chat.[3]
        full_prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )

        # The mlx_lm.generate function produces the model's response.[4]
        response = generate(
            model,
            tokenizer,
            prompt=full_prompt,
            max_tokens=args.max_tokens,
        )

        print(f"Bot: {response}")

        # Add the model's response to the history for the next turn.
        history.append({"role": "assistant", "content": response})

    print("\nExiting chat session.")

if __name__ == "__main__":
    main()
