"""
Qwen3 Implementation from Scratch

This package provides a complete implementation of Qwen3 0.6B model
based on the "Build a Large Language Model From Scratch" repository.
"""

from .model import Qwen3Model, QWEN_CONFIG_06_B
from .tokenizer import Qwen3Tokenizer
from .utils import download_from_huggingface
from .generation import generate_text

__version__ = "1.0.0"
__all__ = [
    "Qwen3Model",
    "QWEN_CONFIG_06_B", 
    "Qwen3Tokenizer",
    "download_from_huggingface",
    "generate_text"
]