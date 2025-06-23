"""
Text generation utilities for Qwen3 models.
"""

import torch
import time


def generate_text(model, tokenizer, prompt, max_new_tokens=150, temperature=0.0, 
                  top_k=None, device=None, verbose=True):
    """
    Generate text using a Qwen3 model and tokenizer.
    
    Args:
        model: Qwen3Model instance
        tokenizer: Qwen3Tokenizer instance
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 = greedy, >1.0 = more random)
        top_k: If set, only sample from top k tokens
        device: Device to run generation on
        verbose: Whether to print generation info
        
    Returns:
        Dictionary with generation results
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Encode the prompt
    input_token_ids = tokenizer.encode(prompt)
    if verbose:
        print(f"Input prompt: {prompt}")
        print(f"Input tokens: {len(input_token_ids)}")
    
    # Convert to tensor and add batch dimension
    idx = torch.tensor(input_token_ids, device=device).unsqueeze(0)
    
    # Track generation time
    start_time = time.time()
    
    # Generate text
    with torch.no_grad():
        output_tokens = generate(
            model=model,
            idx=idx,
            max_new_tokens=max_new_tokens,
            context_size=model.cfg["context_length"],
            temperature=temperature,
            top_k=top_k
        )
    
    generation_time = time.time() - start_time
    
    # Decode the output
    output_text = tokenizer.decode(output_tokens.squeeze(0).tolist())
    
    # Calculate statistics
    total_tokens = len(output_tokens[0])
    new_tokens = total_tokens - len(input_token_ids)
    tokens_per_sec = new_tokens / generation_time if generation_time > 0 else 0
    
    if verbose:
        print(f"Generation completed in {generation_time:.2f} seconds")
        print(f"Generated {new_tokens} new tokens ({tokens_per_sec:.1f} tokens/sec)")
        
        # Memory usage (if CUDA)
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"Peak memory usage: {max_memory:.2f} GB")
    
    return {
        "output_text": output_text,
        "input_tokens": len(input_token_ids),
        "output_tokens": total_tokens,
        "new_tokens": new_tokens,
        "generation_time": generation_time,
        "tokens_per_sec": tokens_per_sec,
        "temperature": temperature,
        "top_k": top_k
    }


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    Core generation function (similar to the one from chapter 5).
    
    Args:
        model: The model to use for generation
        idx: Starting token indices (batch_size, seq_len)
        max_new_tokens: Number of new tokens to generate
        context_size: Maximum context length
        temperature: Sampling temperature
        top_k: Top-k filtering
        eos_id: End-of-sequence token ID (stops generation early if encountered)
        
    Returns:
        Generated token sequences
    """
    for _ in range(max_new_tokens):
        # Crop sequence if it exceeds context size
        idx_cond = idx[:, -context_size:]
        
        # Get model predictions
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]  # Focus on last time step

        # Apply top-k filtering if specified
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, 
                torch.tensor(float('-inf')).to(logits.device), 
                logits
            )

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy sampling
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Check for end-of-sequence
        if eos_id is not None and idx_next.item() == eos_id:
            break

        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def batch_generate(model, tokenizer, prompts, max_new_tokens=150, temperature=0.0, 
                   top_k=None, device=None):
    """
    Generate text for multiple prompts in a batch.
    
    Args:
        model: Qwen3Model instance
        tokenizer: Qwen3Tokenizer instance
        prompts: List of input prompts
        max_new_tokens: Maximum number of tokens to generate per prompt
        temperature: Sampling temperature
        top_k: Top-k filtering
        device: Device to run generation on
        
    Returns:
        List of generation results
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    results = []
    
    print(f"Generating text for {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts):
        print(f"Processing prompt {i+1}/{len(prompts)}")
        
        result = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
            verbose=False
        )
        
        results.append(result)
    
    return results


def interactive_chat(model, tokenizer, device=None, max_new_tokens=150, 
                     temperature=0.7, top_k=50):
    """
    Interactive chat session with the model.
    
    Args:
        model: Qwen3Model instance
        tokenizer: Qwen3Tokenizer instance
        device: Device to run generation on
        max_new_tokens: Maximum tokens per response
        temperature: Sampling temperature
        top_k: Top-k filtering
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    print("Starting interactive chat session (type 'quit' to exit)")
    print(f"Settings: max_tokens={max_new_tokens}, temperature={temperature}, top_k={top_k}")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not user_input:
                continue
            
            print("Assistant: ", end="", flush=True)
            
            result = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=user_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device,
                verbose=False
            )
            
            # Extract just the assistant's response (after the input)
            output_text = result["output_text"]
            
            # Find the assistant's response part
            if "<|im_start|>assistant" in output_text:
                assistant_part = output_text.split("<|im_start|>assistant", 1)[1]
                if assistant_part.startswith("\n"):
                    assistant_part = assistant_part[1:]
                print(assistant_part)
            else:
                print(output_text)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue