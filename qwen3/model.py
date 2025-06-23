"""
Main Qwen3 model implementation.
"""

import torch
import torch.nn as nn
from .config import QWEN_CONFIG_06_B
from .layers import TransformerBlock, RMSNorm
from .positional_encoding import compute_rope_params


class Qwen3Model(nn.Module):
    """
    Qwen3 transformer model implementation.
    
    This implements the Qwen3 architecture with:
    - Token embeddings
    - Multiple transformer blocks with grouped query attention
    - RMSNorm normalization
    - Rotary position embeddings (RoPE)
    - Weight tying between input and output embeddings
    """
    
    def __init__(self, cfg=None):
        super().__init__()
        
        # Use default config if none provided
        if cfg is None:
            cfg = QWEN_CONFIG_06_B
        
        self.cfg = cfg

        # Main model parameters
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        # Stack of transformer blocks
        self.trf_blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg["n_layers"])
        ])

        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Precompute RoPE parameters
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
            
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        
        # Register as buffers (not parameters, but part of model state)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, in_idx):
        """
        Forward pass through the model.
        
        Args:
            in_idx: Input token indices of shape (batch_size, seq_len)
        
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Token embeddings
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        # Create causal mask for attention
        num_tokens = x.shape[1]
        mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), 
            diagonal=1
        )

        # Pass through all transformer blocks
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
            
        # Final normalization
        x = self.final_norm(x)
        
        # Output projection to vocabulary
        logits = self.out_head(x.to(self.cfg["dtype"]))
        
        return logits

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Subtract embedding parameters (they're tied with output layer)
            n_params -= self.tok_emb.weight.numel()
        return n_params

    def estimate_memory_usage(self, dtype=None):
        """
        Estimate memory usage of the model in GB.
        
        Args:
            dtype: Data type to use for estimation (defaults to model's dtype)
        
        Returns:
            Dictionary with memory usage breakdown
        """
        if dtype is None:
            dtype = self.cfg["dtype"]
            
        total_params = sum(p.numel() for p in self.parameters())
        total_buffers = sum(buf.numel() for buf in self.buffers())
        
        # Size in bytes per element
        element_size = torch.tensor(0, dtype=dtype).element_size()
        
        # Calculate memory for parameters, gradients, and buffers
        params_memory = total_params * element_size
        grads_memory = total_params * element_size  # Assuming gradients are stored
        buffers_memory = total_buffers * element_size
        
        total_memory = params_memory + grads_memory + buffers_memory
        
        return {
            "parameters_gb": params_memory / (1024**3),
            "gradients_gb": grads_memory / (1024**3),
            "buffers_gb": buffers_memory / (1024**3),
            "total_gb": total_memory / (1024**3),
            "dtype": str(dtype)
        }
        
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Simple text generation using the model.
        
        Args:
            idx: Starting token indices (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (0.0 = greedy, >1.0 = more random)
            top_k: If set, only sample from top k tokens
        
        Returns:
            Generated token indices including the input
        """
        for _ in range(max_new_tokens):
            # Crop sequence if it exceeds model's context length
            idx_cond = idx[:, -self.cfg["context_length"]:]
            
            # Get predictions
            logits = self(idx_cond)
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
            
            # Apply temperature and sample
            if temperature == 0.0:
                # Greedy sampling
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Probabilistic sampling
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx