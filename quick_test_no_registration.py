"""
Quick test to verify all imports work correctly (No Registration Version)
"""

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    
    try:
        # Test config imports
        from TinyPeLLMConfig import TinyPeLLMConfig
        print("✅ Config imports successful")
        
        # Test model imports
        from TinyPeLLMModel import TinyPeLLMForCausalLM, TinyPeLLMModel
        print("✅ Model imports successful")
        
        # Test tokenizer imports
        from TinyPeLLMTokenizer import TinyPeLLMTokenizer
        print("✅ Tokenizer imports successful")
        
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
    """Test basic functionality without auto registration"""
    print("\nTesting basic functionality...")
    
    try:
        # Import classes directly (no auto registration)
        from TinyPeLLMConfig import TinyPeLLMConfig
        from TinyPeLLMModel import TinyPeLLMForCausalLM
        from TinyPeLLMTokenizer import TinyPeLLMTokenizer
        
        print("✅ Direct imports successful")
        
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
        
        # Test text generation
        generated = model.generate(
            inputs.input_ids,
            max_length=20,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"✅ Text generation successful: '{generated_text}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_usage():
    """Test pipeline usage without auto registration"""
    print("\nTesting pipeline usage...")
    
    try:
        from TinyPeLLMConfig import TinyPeLLMConfig
        from TinyPeLLMModel import TinyPeLLMForCausalLM
        from TinyPeLLMTokenizer import TinyPeLLMTokenizer
        from transformers import pipeline
        
        # Create model and tokenizer
        config = TinyPeLLMConfig(vocab_size=3001, hidden_size=128)
        tokenizer = TinyPeLLMTokenizer("tinypellm.model")
        model = TinyPeLLMForCausalLM(config)
        
        # Create pipeline manually
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1  # Use CPU
        )
        
        # Test pipeline
        result = pipe(
            "Hello, how are you?",
            max_length=30,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True
        )
        
        print(f"✅ Pipeline test successful: {result[0]['generated_text']}")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run quick tests"""
    print("🚀 Quick Test for TinyPeLLM (No Registration)")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test basic functionality
    if imports_ok:
        functionality_ok = test_basic_functionality()
        pipeline_ok = test_pipeline_usage()
    else:
        functionality_ok = False
        pipeline_ok = False
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 Test Summary:")
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Basic Functionality: {'✅ PASS' if functionality_ok else '❌ FAIL'}")
    print(f"Pipeline Usage: {'✅ PASS' if pipeline_ok else '❌ FAIL'}")
    
    if imports_ok and functionality_ok and pipeline_ok:
        print("\n🎉 All tests passed! TinyPeLLM is ready to use.")
        print("\nNext steps:")
        print("1. Use direct imports instead of auto registration")
        print("2. Create models and tokenizers manually")
        print("3. Use pipeline with explicit model and tokenizer")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    return imports_ok and functionality_ok and pipeline_ok


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 