# src/mlx_grpo/config.py

from dataclasses import dataclass

@dataclass
class GRPOConfig:
    """
    Configuration for the GRPOTrainer.
    """
    iters: int = 200
    batch_size: int = 2
    group_size: int = 4
    learning_rate: float = 1e-5
    epsilon: float = 0.2
    beta: float = 0.01
    update_every: int = 10
    max_ans_len: int = 4
    # Use -100 as the ignore_index for the loss function, a standard in Hugging Face and PyTorch
    ignore_index: int = -100
