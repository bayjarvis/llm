# MLX-GRPO: Group Relative Policy Optimization for MLX

An implementation of Group Relative Policy Optimization (GRPO) in Apple's MLX framework for fine-tuning language models on Apple Silicon.

## Installation

```bash
# Clone the repository
git clone https://github.com/bayjarvis/llm/mlx-grpo
cd mlx-grpo

# Install the package
pip install -e .
```

This will automatically install all required dependencies including:
- `mlx` - Apple's ML framework
- `mlx-lm` - MLX language model utilities
- `transformers` - For tokenizers
- `numpy` and `tqdm`

## Quick Start

### Training

Train a model using GRPO with the default configuration:

```bash
mlx-grpo-train
```

The training script uses `Qwen/Qwen3-0.6B` as the default model and runs for 200 iterations with a dummy dataset.

### Chat Interface

After training, you can chat with the model:

```bash
# Chat with base model only
mlx-grpo-chat --model Qwen/Qwen3-0.6B --adapter-file ""

# Chat with trained adapter (default behavior)
mlx-grpo-chat --model Qwen/Qwen3-0.6B
```

The system automatically loads the `grpo_adapter/` directory by default and applies the trained weights to the base model.

### Chat Options

- `--model`: Specify the base model (e.g., `Qwen/Qwen3-0.6B`)
- `--adapter-file`: Path to trained adapter directory (default: `grpo_adapter`)
- `--max-tokens`: Maximum tokens to generate (default: 250)
- `--temp`: Temperature for generation (default: 0.7)

## Example Usage

```bash
# Start training
mlx-grpo-train

# After training completes, start chatting
mlx-grpo-chat --model Qwen/Qwen3-0.6B

# In the chat interface:
You: What is the capital of France?
Assistant: The capital of France is Paris.

You: exit
```

## Configuration

The training configuration can be modified in `src/mlx_grpo/run.py`:

```python
config = GRPOConfig(
    iters=200,           # Number of training iterations
    batch_size=2,        # Batch size
    group_size=4,        # Group size for GRPO
    learning_rate=1e-5,  # Learning rate
    epsilon=0.2,         # PPO epsilon
    beta=0.01,           # KL regularization
    update_every=10,     # Sync old model weights every N steps
    max_ans_len=4        # Maximum answer length
)
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+
- MLX framework

## Adapter System

### Space Efficiency
- **Adapter format**: Saves only parameter differences (deltas) from base model
- **Typical savings**: 95-99% reduction in storage (5-50MB vs 1-20GB)
- **Current demo**: ~50% savings (594MB adapter vs 1.2GB full model)

### Custom Adapter Loading
- Implements custom adapter loading to bypass MLX-LM format restrictions
- Automatically applies adapter deltas to base model parameters
- Graceful fallback to base model if adapter loading fails

## Model Support

Currently tested with:
- `Qwen/Qwen3-0.6B` (recommended)
- Other Qwen models
- Compatible HuggingFace models

## Technical Details

This implementation:
- Uses real MLX models (Qwen3-0.6B) instead of placeholders
- Implements GRPO (Group Relative Policy Optimization) algorithm
- Saves adapters in safetensors format for security and performance
- Includes custom adapter loading for maximum compatibility