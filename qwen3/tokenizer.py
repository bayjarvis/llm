"""
Qwen3 tokenizer implementation with support for chat formatting and reasoning models.
"""

from pathlib import Path
from .utils import download_from_huggingface


class Qwen3Tokenizer:
    """
    Qwen3 tokenizer with support for chat formatting and reasoning capabilities.
    
    This tokenizer handles:
    - Basic encoding/decoding of text
    - Chat message formatting with special tokens
    - Support for reasoning models with thinking tokens
    - Automatic downloading of tokenizer files
    """
    
    def __init__(self, tokenizer_file_path="tokenizer.json", repo_id=None, 
                 add_generation_prompt=False, add_thinking=False):
        """
        Initialize the Qwen3 tokenizer.
        
        Args:
            tokenizer_file_path: Path to the tokenizer.json file
            repo_id: Hugging Face repository ID for downloading tokenizer
            add_generation_prompt: Whether to add generation prompt for chat
            add_thinking: Whether to add thinking tokens for reasoning models
        """
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError(
                "The 'tokenizers' library is required. Install it with: pip install tokenizers"
            )
            
        self.tokenizer_file_path = tokenizer_file_path

        # Validate reasoning model settings
        if add_generation_prompt != add_thinking:
            raise ValueError(
                "Currently only add_generation_prompt==add_thinking settings are supported"
            )

        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        # Download tokenizer if it doesn't exist locally
        tokenizer_file_path_obj = Path(tokenizer_file_path)
        if not tokenizer_file_path_obj.is_file() and repo_id is not None:
            download_from_huggingface(
                repo_id=repo_id,
                filename=str(tokenizer_file_path_obj.name),
                local_dir=str(tokenizer_file_path_obj.parent.name)
            )
            
        # Load the tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_file_path)

    def encode(self, prompt):
        """
        Encode a text prompt into token IDs.
        
        Args:
            prompt: Text string to encode
            
        Returns:
            List of token IDs
        """
        # Format as chat message if needed
        if self.add_generation_prompt or self.add_thinking:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.format_qwen_chat(
                messages,
                add_generation_prompt=self.add_generation_prompt,
                add_thinking=self.add_thinking
            )
        else:
            formatted_prompt = prompt
            
        return self.tokenizer.encode(formatted_prompt).ids

    def decode(self, token_ids, skip_special_tokens=False):
        """
        Decode token IDs back into text.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @staticmethod
    def format_qwen_chat(messages, add_generation_prompt=False, add_thinking=False):
        """
        Format messages using Qwen's chat template.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            add_generation_prompt: Whether to add the assistant generation prompt
            add_thinking: Whether to add thinking tokens for reasoning models
            
        Returns:
            Formatted prompt string
        """
        prompt = ""
        
        # Add each message with proper formatting
        for msg in messages:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            
        # Add generation prompt if requested
        if add_generation_prompt:
            prompt += "<|im_start|>assistant"
            if not add_thinking:
                # For base models, add empty thinking block
                prompt += "<|think>\n\n<|/think>\n\n"
            else:
                # For reasoning models, let the model generate thinking content
                prompt += "\n"
                
        return prompt

    def get_special_tokens(self):
        """
        Get information about special tokens used by the tokenizer.
        
        Returns:
            Dictionary with special token information
        """
        special_tokens = {
            "im_start": "<|im_start|>",
            "im_end": "<|im_end|>", 
            "think_start": "<|think|>",
            "think_end": "<|/think|>",
            "endoftext": "<|endoftext|>"
        }
        
        # Try to get token IDs for special tokens
        token_ids = {}
        for name, token in special_tokens.items():
            try:
                token_ids[name] = self.tokenizer.encode(token).ids[0]
            except:
                token_ids[name] = None
                
        return {
            "tokens": special_tokens,
            "token_ids": token_ids
        }

    def count_tokens(self, text):
        """
        Count the number of tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.encode(text))
        
    def validate_tokenizer(self):
        """
        Validate that the tokenizer is working correctly.
        
        Returns:
            Dictionary with validation results
        """
        test_text = "Hello, world!"
        
        try:
            # Test basic encode/decode
            tokens = self.encode(test_text)
            decoded = self.decode(tokens)
            
            # Test chat formatting
            chat_messages = [{"role": "user", "content": "Test message"}]
            chat_formatted = self.format_qwen_chat(chat_messages)
            chat_tokens = self.tokenizer.encode(chat_formatted).ids
            
            return {
                "status": "success",
                "basic_encoding": len(tokens) > 0,
                "basic_decoding": len(decoded) > 0,
                "chat_formatting": len(chat_tokens) > 0,
                "vocab_size": self.tokenizer.get_vocab_size(),
                "test_tokens": tokens[:10]  # First 10 tokens as example
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }