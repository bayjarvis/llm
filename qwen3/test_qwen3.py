"""
Test script for Qwen3 implementation.

This script tests various components of the Qwen3 implementation
to ensure everything works correctly.
"""

import torch
import sys
from pathlib import Path

# Add the parent directory to the path so we can import qwen3
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from qwen3.config import QWEN_CONFIG_06_B, get_qwen_config
        from qwen3.model import Qwen3Model
        from qwen3.layers import RMSNorm, FeedForward, GroupedQueryAttention, TransformerBlock
        from qwen3.positional_encoding import compute_rope_params, apply_rope
        from qwen3.utils import get_device, validate_config, print_model_info
        from qwen3.generation import generate
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test configuration validation."""
    print("Testing configuration...")
    
    try:
        from qwen3.config import QWEN_CONFIG_06_B, get_qwen_config
        from qwen3.utils import validate_config
        
        # Test default config
        config = get_qwen_config("0.6B")
        validate_config(config)
        
        # Test that config has required keys
        required_keys = ["vocab_size", "context_length", "emb_dim", "n_heads", "n_layers"]
        for key in required_keys:
            assert key in config, f"Missing key: {key}"
        
        print("✓ Configuration tests passed")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_layers():
    """Test individual layer components."""
    print("Testing layers...")
    
    try:
        from qwen3.layers import RMSNorm, FeedForward, GroupedQueryAttention
        from qwen3.config import QWEN_CONFIG_06_B
        
        # Test RMSNorm
        norm = RMSNorm(768)
        x = torch.randn(2, 10, 768)
        out = norm(x)
        assert out.shape == x.shape, "RMSNorm output shape mismatch"
        
        # Test FeedForward
        ff = FeedForward(QWEN_CONFIG_06_B)
        x = torch.randn(2, 10, 1024, dtype=torch.bfloat16)
        out = ff(x)
        assert out.shape == x.shape, "FeedForward output shape mismatch"
        
        # Test GroupedQueryAttention (simplified)
        gqa = GroupedQueryAttention(
            d_in=1024, 
            num_heads=16, 
            num_kv_groups=8, 
            head_dim=64,
            dtype=torch.bfloat16
        )
        x = torch.randn(2, 10, 1024, dtype=torch.bfloat16)
        mask = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
        
        # Create dummy cos/sin for RoPE
        cos = torch.randn(10, 64)
        sin = torch.randn(10, 64)
        
        out = gqa(x, mask, cos, sin)
        assert out.shape == x.shape, "GroupedQueryAttention output shape mismatch"
        
        print("✓ Layer tests passed")
        return True
    except Exception as e:
        print(f"✗ Layer test failed: {e}")
        return False

def test_rope():
    """Test Rotary Position Embedding."""
    print("Testing RoPE...")
    
    try:
        from qwen3.positional_encoding import compute_rope_params, apply_rope
        
        # Test RoPE parameter computation
        head_dim = 64
        context_length = 100
        cos, sin = compute_rope_params(head_dim, context_length=context_length)
        
        assert cos.shape == (context_length, head_dim), "RoPE cos shape mismatch"
        assert sin.shape == (context_length, head_dim), "RoPE sin shape mismatch"
        
        # Test RoPE application
        x = torch.randn(2, 8, 50, head_dim)  # (batch, heads, seq_len, head_dim)
        x_rope = apply_rope(x, cos, sin)
        
        assert x_rope.shape == x.shape, "RoPE output shape mismatch"
        assert x_rope.dtype == x.dtype, "RoPE dtype mismatch"
        
        print("✓ RoPE tests passed")
        return True
    except Exception as e:
        print(f"✗ RoPE test failed: {e}")
        return False

def test_model():
    """Test the complete model."""
    print("Testing Qwen3Model...")
    
    try:
        from qwen3.model import Qwen3Model
        from qwen3.config import QWEN_CONFIG_06_B
        
        # Create a smaller config for testing
        test_config = QWEN_CONFIG_06_B.copy()
        test_config.update({
            "n_layers": 2,  # Reduce layers for faster testing
            "context_length": 100,  # Reduce context length
            "emb_dim": 256,  # Reduce embedding dimension
            "hidden_dim": 512,  # Reduce FFN dimension
            "n_heads": 4,  # Reduce attention heads
            "n_kv_groups": 2,  # Reduce KV groups
            "head_dim": 64,  # Set head dimension
            "vocab_size": 1000,  # Reduce vocabulary
            "dtype": torch.float32  # Use float32 for testing
        })
        
        # Test model creation
        model = Qwen3Model(test_config)
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 10))  # (batch_size, seq_len)
        logits = model(input_ids)
        
        expected_shape = (2, 10, 1000)  # (batch_size, seq_len, vocab_size)
        assert logits.shape == expected_shape, f"Model output shape mismatch: {logits.shape} vs {expected_shape}"
        
        # Test parameter counting
        num_params = model.get_num_params()
        assert num_params > 0, "Model should have parameters"
        
        # Test memory estimation
        memory_info = model.estimate_memory_usage()
        assert "total_gb" in memory_info, "Memory estimation should include total_gb"
        
        print("✓ Model tests passed")
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_generation():
    """Test text generation (without pretrained weights)."""
    print("Testing generation...")
    
    try:
        from qwen3.model import Qwen3Model
        from qwen3.generation import generate
        from qwen3.config import QWEN_CONFIG_06_B
        
        # Create a small model for testing
        test_config = QWEN_CONFIG_06_B.copy()
        test_config.update({
            "n_layers": 2,
            "context_length": 50,
            "emb_dim": 128,
            "hidden_dim": 256,
            "n_heads": 4,
            "n_kv_groups": 2,
            "head_dim": 32,
            "vocab_size": 100,
            "dtype": torch.float32
        })
        
        model = Qwen3Model(test_config)
        model.eval()
        
        # Test generation
        input_ids = torch.randint(0, 100, (1, 5))  # Start with 5 tokens
        
        output = generate(
            model=model,
            idx=input_ids,
            max_new_tokens=10,
            context_size=test_config["context_length"],
            temperature=1.0,
            top_k=10
        )
        
        # Check that we generated new tokens
        assert output.shape[1] == input_ids.shape[1] + 10, "Should generate 10 new tokens"
        
        print("✓ Generation tests passed")
        return True
    except Exception as e:
        print(f"✗ Generation test failed: {e}")
        return False

def test_tokenizer_mock():
    """Test tokenizer functionality (without actual tokenizer file)."""
    print("Testing tokenizer (mock)...")
    
    try:
        from qwen3.tokenizer import Qwen3Tokenizer
        
        # Test chat formatting (static method, doesn't need tokenizer file)
        messages = [
            {"role": "user", "content": "Hello, world!"}
        ]
        
        # Test basic formatting
        formatted = Qwen3Tokenizer.format_qwen_chat(messages)
        assert "<|im_start|>user" in formatted, "Should contain user start token"
        assert "Hello, world!" in formatted, "Should contain message content"
        assert "<|im_end|>" in formatted, "Should contain end token"
        
        # Test with generation prompt
        formatted_with_prompt = Qwen3Tokenizer.format_qwen_chat(
            messages, 
            add_generation_prompt=True, 
            add_thinking=False
        )
        assert "<|im_start|>assistant" in formatted_with_prompt, "Should contain assistant start"
        assert "<|think|>" in formatted_with_prompt, "Should contain thinking tokens"
        
        print("✓ Tokenizer tests passed")
        return True
    except Exception as e:
        print(f"✗ Tokenizer test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 50)
    print("Running Qwen3 Implementation Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_layers,
        test_rope,
        test_model,
        test_generation,
        test_tokenizer_mock
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Qwen3 implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("=" * 50)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)