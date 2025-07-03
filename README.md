# LLM Training and Fine-tuning Projects

This repository contains various LLM training implementations using different optimization approaches.

## Projects by Training Approach

### Self-Supervised Alignment

* **[MLX-GRPO: Group Relative Policy Optimization](https://github.com/bayjarvis/llm/tree/main/mlx-grpo)** - Complete GRPO implementation for Apple Silicon using MLX framework with Qwen3-0.6B model support. Uses group comparisons for alignment without requiring human feedback data.

### Human Feedback-Based Training

* **[Harnessing Zephyr's Breeze: DPO Training on Mistral-7B-GPTQ](https://github.com/bayjarvis/llm/tree/main/mistral/dpo)** - Direct Preference Optimization for language model alignment using human preference datasets on quantized models.

### Supervised Fine-tuning

* **[Fine-tuning Zephyr 7B GPTQ with 4-Bit Quantization](https://github.com/bayjarvis/llm/tree/main/zephyr/finetune_gptq)** - Custom data fine-tuning with 4-bit quantization for efficient inference and deployment.
