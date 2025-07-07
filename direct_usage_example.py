"""
Direct Usage Example for TinyPeLLM
Shows how to use the model without auto registration
"""

import torch
from transformers import pipeline

# Direct imports (no auto registration needed)
from TinyPeLLMConfig import TinyPeLLMConfig
from TinyPeLLMModel import TinyPeLLMForCausalLM
from TinyPeLLMTokenizer import TinyPeLLMTokenizer


def example_1_basic_usage():
    """Example 1: Basic model usage"""
    print("=== Example 1: Basic Model Usage ===")
    
    # Create config
    config = TinyPeLLMConfig(
        vocab_size=3001,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4
    )
    
    # Create tokenizer
    tokenizer = TinyPeLLMTokenizer("tinypellm.model")
    
    # Create model
    model = TinyPeLLMForCausalLM(config)
    
    # Test forward pass
    text = "Hello, how are you?"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Input: {text}")
    print(f"Output shape: {outputs.logits.shape}")
    print("✅ Basic usage successful!")


def example_2_text_generation():
    """Example 2: Text generation"""
    print("\n=== Example 2: Text Generation ===")
    
    # Create model and tokenizer
    config = TinyPeLLMConfig(vocab_size=3001, hidden_size=128)
    tokenizer = TinyPeLLMTokenizer("tinypellm.model")
    model = TinyPeLLMForCausalLM(config)
    
    # Generate text
    prompt = "The weather is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        generated = model.generate(
            inputs.input_ids,
            max_length=50,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("✅ Text generation successful!")


def example_3_pipeline_usage():
    """Example 3: Pipeline usage"""
    print("\n=== Example 3: Pipeline Usage ===")
    
    # Create model and tokenizer
    config = TinyPeLLMConfig(vocab_size=3001, hidden_size=128)
    tokenizer = TinyPeLLMTokenizer("tinypellm.model")
    model = TinyPeLLMForCausalLM(config)
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1  # Use CPU
    )
    
    # Generate text with pipeline
    result = pipe(
        "Hello, how are you?",
        max_length=30,
        num_return_sequences=1,
        temperature=0.8,
        do_sample=True
    )
    
    print(f"Pipeline result: {result[0]['generated_text']}")
    print("✅ Pipeline usage successful!")


def example_4_load_pretrained():
    """Example 4: Load pretrained model"""
    print("\n=== Example 4: Load Pretrained Model ===")
    
    try:
        # Try to load existing model weights
        config = TinyPeLLMConfig(vocab_size=3001, hidden_size=128)
        tokenizer = TinyPeLLMTokenizer("tinypellm.model")
        model = TinyPeLLMForCausalLM(config)
        
        # Load pretrained weights if available
        if torch.cuda.is_available():
            state_dict = torch.load("tinypellm/tinypellm_model.pt", map_location='cpu')
        else:
            state_dict = torch.load("tinypellm/tinypellm_model.pt", map_location='cpu')
        
        model.load_state_dict(state_dict)
        print("✅ Pretrained model loaded successfully!")
        
        # Test generation
        inputs = tokenizer("Hello", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=20,
                temperature=0.8,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated with pretrained model: {generated_text}")
        
    except Exception as e:
        print(f"⚠️ Could not load pretrained model: {e}")
        print("Using untrained model instead...")


def main():
    """Run all examples"""
    print("🚀 TinyPeLLM Direct Usage Examples")
    print("=" * 50)
    
    try:
        example_1_basic_usage()
        example_2_text_generation()
        example_3_pipeline_usage()
        example_4_load_pretrained()
        
        print("\n🎉 All examples completed successfully!")
        print("\n📝 Summary:")
        print("- Use direct imports instead of auto registration")
        print("- Create models and tokenizers manually")
        print("- Use pipeline with explicit model and tokenizer")
        print("- Load pretrained weights if available")
        
    except Exception as e:
        print(f"\n❌ Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 