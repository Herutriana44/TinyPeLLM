"""
Quick test to verify all imports work correctly (Simple Version)
"""

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    
    try:
        # Test config imports
        from TinyPeLLMConfig import TinyPeLLMConfig
        print("✅ Config imports successful")
        
        # Test model imports
        from TinyPeLLMModel_clean import TinyPeLLMForCausalLM, TinyPeLLMModel
        print("✅ Model imports successful")
        
        # Test tokenizer imports
        from TinyPeLLMTokenizer import TinyPeLLMTokenizer
        print("✅ Tokenizer imports successful")
        
        # Test pipeline imports (simplified)
        from TinyPeLLMPipeline_simple import (
            register_tiny_pellm,
            TinyPeLLMTrainer,
            TinyPeLLMPipeline,
            create_tiny_pellm_pipeline
        )
        print("✅ Pipeline imports successful")
        
        # Test transformers imports
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        print("✅ Transformers imports successful")
        
        # Test torch
        import torch
        print("✅ PyTorch import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        # Import classes
        from TinyPeLLMConfig import TinyPeLLMConfig
        from TinyPeLLMModel_clean import TinyPeLLMForCausalLM
        from TinyPeLLMTokenizer import TinyPeLLMTokenizer
        from TinyPeLLMPipeline_simple import register_tiny_pellm
        
        # Register model
        register_tiny_pellm()
        print("✅ Model registration successful")
        
        # Create config
        config = TinyPeLLMConfig(vocab_size=3001, hidden_size=128)
        print("✅ Config creation successful")
        
        # Create tokenizer
        tokenizer = TinyPeLLMTokenizer("tinypellm.model")
        print("✅ Tokenizer creation successful")
        
        # Create model
        model = TinyPeLLMForCausalLM(config)
        print("✅ Model creation successful")
        
        # Test basic forward pass
        import torch
        inputs = tokenizer("Hello", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✅ Forward pass successful: {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run quick tests"""
    print("🚀 Quick Test for TinyPeLLM (Simple)")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test basic functionality
    if imports_ok:
        functionality_ok = test_basic_functionality()
    else:
        functionality_ok = False
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 Test Summary:")
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Basic Functionality: {'✅ PASS' if functionality_ok else '❌ FAIL'}")
    
    if imports_ok and functionality_ok:
        print("\n🎉 All tests passed! TinyPeLLM is ready to use.")
        print("\nNext steps:")
        print("1. Run: python demo.py")
        print("2. Run: python test_tiny_pellm.py")
        print("3. Check: python example_usage.py")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    return imports_ok and functionality_ok


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 