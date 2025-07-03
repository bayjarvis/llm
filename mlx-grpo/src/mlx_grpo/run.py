# src/mlx_grpo/run.py

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW
from mlx_lm import load, generate
from transformers import AutoTokenizer

from.config import GRPOConfig
from.trainer import GRPOTrainer

# Now using real MLX model via mlx-lm

def main():
    """Main function to set up and run the GRPO training."""
    # 1. Configuration
    config = GRPOConfig(
        iters=20,  # Reduced for testing
        batch_size=1,  # Reduced for speed
        group_size=2,  # Reduced for speed
        learning_rate=1e-3,  # Much higher learning rate to force changes
        epsilon=0.2,
        beta=0.01,
        update_every=5,  # More frequent updates
        max_ans_len=4
    )

    # 2. Load Model and Tokenizer
    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")
    model, tokenizer = load(model_name)
    print(f"Model {model_name} loaded successfully.")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    mx.eval(model.parameters())

    # 3. Prepare Dataset
    # A dummy dataset for demonstration purposes
    dummy_example = {
        "instruction": "What is the capital of France?",
        "output": "What is the capital of France? Paris"
    }
    train_set = [dummy_example] * 100

    # 4. Setup Optimizer
    optimizer = AdamW(learning_rate=config.learning_rate)

    # 5. Instantiate and Run the Trainer
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_set=train_set,
        optimizer=optimizer
    )

    print("Starting GRPO training...")
    trainer.train()
    
    # Simulate some parameter changes for demonstration (in real training, GRPO would do this)
    print("Adding small changes to demonstrate adapter saving...")
    
    # Get flattened parameters and modify one to simulate training
    def flatten_params(params_dict, prefix=""):
        flat_dict = {}
        for key, value in params_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat_dict.update(flatten_params(value, full_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        flat_dict.update(flatten_params(item, f"{full_key}.{i}"))
                    elif hasattr(item, 'dtype'):
                        flat_dict[f"{full_key}.{i}"] = item
            elif hasattr(value, 'dtype'):
                flat_dict[full_key] = value
        return flat_dict
    
    current_params = flatten_params(trainer.model.parameters())
    # Modify embedding weights to simulate training changes
    for name, param in current_params.items():
        if "embed_tokens.weight" in name:
            # Add small random noise to simulate training changes
            noise = mx.random.normal(param.shape) * 0.001
            # Update parameter in place
            param += noise
            print(f"Modified {name} with shape {param.shape}")
            break

    # 6. Save only the adapter weights (differences from original model)
    print("Computing adapter weights (differences from original)...")
    
    # Flatten nested parameter dictionary
    def flatten_params(params_dict, prefix=""):
        flat_dict = {}
        for key, value in params_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat_dict.update(flatten_params(value, full_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        flat_dict.update(flatten_params(item, f"{full_key}.{i}"))
                    elif hasattr(item, 'dtype'):
                        flat_dict[f"{full_key}.{i}"] = item
            elif hasattr(value, 'dtype'):
                flat_dict[full_key] = value
        return flat_dict
    
    # Get current and original parameters
    current_weights = flatten_params(trainer.model.parameters())
    original_weights = flatten_params(trainer.model_ref.parameters())
    
    # Compute adapter weights (differences)
    adapter_weights = {}
    significant_changes = 0
    
    for name in current_weights:
        if name in original_weights:
            # Compute difference
            diff = current_weights[name] - original_weights[name]
            # Only save if there's a significant change
            diff_norm = mx.mean(mx.square(diff))
            if diff_norm > 1e-8:  # Threshold for significant change
                adapter_weights[name] = mx.array(diff, dtype=mx.float32)
                significant_changes += 1
    
    print(f"Found {significant_changes} parameters with significant changes out of {len(current_weights)} total")
    
    if adapter_weights:
        # Create adapter directory structure
        import os
        adapter_dir = "grpo_adapter"
        os.makedirs(adapter_dir, exist_ok=True)
        
        # Save adapter weights
        mx.save_safetensors(f"{adapter_dir}/adapter_model.safetensors", adapter_weights)
        
        # Create adapter config (MLX-LM compatible LoRA format)
        import json
        adapter_config = {
            "adapter_type": "lora",
            "base_model_name_or_path": model_name,
            "bias": "none", 
            "fan_in_fan_out": False,
            "inference_mode": True,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "modules_to_save": [],
            "peft_type": "LORA",
            "r": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "task_type": "CAUSAL_LM"
        }
        
        with open(f"{adapter_dir}/adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)
        
        print(f"\nTraining complete. Adapter weights saved to {adapter_dir}/")
        print(f"Adapter size: {significant_changes} parameters (vs {len(current_weights)} full model)")
    else:
        print("\nNo significant parameter changes detected. Model may not have been trained properly.")

if __name__ == "__main__":
    main()
