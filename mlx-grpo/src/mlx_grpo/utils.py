# src/mlx_grpo/utils.py

import mlx.core as mx
import mlx.nn as nn

def generate(model, tokenizer, prompt_tokens, max_tokens=100):
    """
    Generate text from a model given a prompt.
    """
    prompt = mx.array(prompt_tokens)
    while True:
        logits = model(prompt[None])[:, -1, :]
        y = mx.argmax(logits, axis=-1)
        prompt = mx.concatenate([prompt, y])
        if len(prompt) - len(prompt_tokens) >= max_tokens or y.item() == tokenizer.eos_token_id:
            break
    
    response_tokens = prompt[len(prompt_tokens):].tolist()
    return tokenizer.decode(response_tokens)

def pad_sequences(sequences, pad_value, max_len=None):
    """
    Pads a list of sequences to the same length.
    """
    if max_len is None:
        max_len = max(len(s) for s in sequences)
    
    padded_sequences = []
    for s in sequences:
        padding_len = max_len - len(s)
        padded_s = mx.concatenate([s, mx.array([pad_value] * padding_len, dtype=s.dtype)])
        padded_sequences.append(padded_s)
        
    return mx.stack(padded_sequences)

def calculate_log_probs(model: nn.Module, sequences: mx.array, labels: mx.array):
    """
    Calculates the log probabilities of the generated answer tokens.
    Uses a standard approach where prompt tokens in the 'labels' tensor
    are masked out with an ignore_index.
    """
    logits = model(sequences)  # (batch, seq_len, vocab_size)
    log_probs_full = nn.log_softmax(logits, axis=-1)

    # Create a mask for valid (non-ignored) labels
    mask = (labels!= -100)

    # Set ignored labels to 0 to avoid index errors; they will be masked out.
    labels_for_gather = mx.where(mask, labels, 0)[:, :, None]

    # Gather the log probs for the target tokens
    selected_log_probs = mx.take_along_axis(log_probs_full, labels_for_gather, axis=-1).squeeze(-1)

    # Apply the mask and sum over the sequence dimension
    masked_log_probs = selected_log_probs * mask
    return mx.sum(masked_log_probs, axis=-1)
