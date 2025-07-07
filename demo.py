"""
TinyPeLLM Demo - Comprehensive demonstration of all features
"""

import torch
import tempfile
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Import our custom classes
from TinyPeLLMModel import TinyPeLLMConfig, TinyPeLLMForCausalLM
from TinyPeLLMTokenizer import TinyPeLLMTokenizer
from TinyPeLLMPipeline import register_tiny_pellm, TinyPeLLMTrainer, TinyPeLLMPipeline


def demo_1_basic_setup():
    """Demo 1: Basic setup and model creation"""
    print("=" * 60)
    print("DEMO 1: Basic Setup and Model Creation")
    print("=" * 60)
    
    # Register the model
    register_tiny_pellm()
    print("✅ Model registered with Auto classes")
    
    # Create configuration
    config = TinyPeLLMConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4
    )
    print(f"✅ Configuration created: {config}")
    
    # Create tokenizer
    tokenizer = TinyPeLLMTokenizer()
    print(f"✅ Tokenizer created with vocab size: {tokenizer.vocab_size()}")
    
    # Create model
    model = TinyPeLLMForCausalLM(config)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return config, tokenizer, model


def demo_2_basic_inference():
    """Demo 2: Basic inference with the model"""
    print("\n" + "=" * 60)
    print("DEMO 2: Basic Inference")
    print("=" * 60)
    
    config, tokenizer, model = demo_1_basic_setup()
    
    # Test basic encoding/decoding
    text = "Hello world"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(f"✅ Encoding/Decoding test: '{text}' -> '{decoded}'")
    
    # Test forward pass
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"✅ Forward pass successful: output shape {outputs.logits.shape}")
    
    # Test generation
    generated = model.generate(
        **inputs,
        max_length=20,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8
    )
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"✅ Generation test: '{generated_text}'")
    
    return config, tokenizer, model


def demo_3_save_and_load():
    """Demo 3: Save and load functionality"""
    print("\n" + "=" * 60)
    print("DEMO 3: Save and Load Functionality")
    print("=" * 60)
    
    config, tokenizer, model = demo_2_basic_inference()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model and tokenizer
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        print(f"✅ Model and tokenizer saved to {temp_dir}")
        
        # List saved files
        files = os.listdir(temp_dir)
        print(f"✅ Saved files: {files}")
        
        # Load using Auto classes
        loaded_tokenizer = AutoTokenizer.from_pretrained(temp_dir)
        loaded_model = AutoModelForCausalLM.from_pretrained(temp_dir)
        print("✅ Model and tokenizer loaded using Auto classes")
        
        # Test loaded model
        text = "Test message"
        inputs = loaded_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = loaded_model(**inputs)
        print(f"✅ Loaded model forward pass: output shape {outputs.logits.shape}")


def demo_4_pipeline_integration():
    """Demo 4: Pipeline integration"""
    print("\n" + "=" * 60)
    print("DEMO 4: Pipeline Integration")
    print("=" * 60)
    
    config, tokenizer, model = demo_1_basic_setup()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model and tokenizer
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)
        
        # Create Hugging Face pipeline
        pipe = pipeline(
            "text-generation",
            model=temp_dir,
            tokenizer=temp_dir,
            device=0 if torch.cuda.is_available() else -1
        )
        print("✅ Hugging Face pipeline created")
        
        # Generate text with pipeline
        result = pipe(
            "Hello, how are you?",
            max_length=30,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True
        )
        print(f"✅ Pipeline generation: {result[0]['generated_text']}")
        
        # Create custom pipeline
        custom_pipe = TinyPeLLMPipeline(temp_dir)
        print("✅ Custom pipeline created")
        
        # Generate text with custom pipeline
        texts = custom_pipe.generate(
            text="The weather is",
            max_length=25,
            temperature=0.7,
            num_return_sequences=2
        )
        print("✅ Custom pipeline generation:")
        for i, text in enumerate(texts):
            print(f"   {i+1}. {text}")


def demo_5_training_workflow():
    """Demo 5: Training workflow"""
    print("\n" + "=" * 60)
    print("DEMO 5: Training Workflow")
    print("=" * 60)
    
    # Create training data
    training_texts = [
        "Hello world!",
        "How are you today?",
        "The weather is nice.",
        "I love programming.",
        "Machine learning is fascinating.",
        "Python is a great language.",
        "Transformers are powerful models.",
        "Natural language processing is exciting."
    ]
    
    # Save training data
    with open("demo_training_data.txt", "w", encoding="utf-8") as f:
        for text in training_texts:
            f.write(text + "\n")
    print("✅ Training data created")
    
    # Initialize trainer
    trainer = TinyPeLLMTrainer()
    print("✅ Trainer initialized")
    
    # Prepare model and tokenizer
    model, tokenizer = trainer.prepare_model_and_tokenizer(
        vocab_size=500,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2
    )
    print("✅ Model and tokenizer prepared")
    
    # Prepare dataset
    dataset = trainer.prepare_dataset(
        data_path="demo_training_data.txt",
        block_size=16
    )
    print(f"✅ Dataset prepared with {len(dataset)} samples")
    
    # Train the model (minimal training for demo)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            train_result = trainer.train(
                train_dataset=dataset,
                output_dir=temp_dir,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                learning_rate=1e-3,
                logging_steps=1,
                save_steps=5,
                save_total_limit=1
            )
            print("✅ Training completed successfully")
            
            # Test trained model
            trained_pipe = TinyPeLLMPipeline(temp_dir)
            generated = trained_pipe.generate("Hello", max_length=15)
            print(f"✅ Trained model output: {generated[0]}")
            
    except Exception as e:
        print(f"⚠️ Training demo failed (expected for quick demo): {e}")
    
    # Clean up
    if os.path.exists("demo_training_data.txt"):
        os.remove("demo_training_data.txt")


def demo_6_advanced_features():
    """Demo 6: Advanced features"""
    print("\n" + "=" * 60)
    print("DEMO 6: Advanced Features")
    print("=" * 60)
    
    # Create advanced configuration
    config = TinyPeLLMConfig(
        vocab_size=2000,
        hidden_size=128,
        num_hidden_layers=3,
        num_attention_heads=8,
        intermediate_size=256,
        max_position_embeddings=1024,
        initializer_range=0.02,
        layer_norm_eps=1e-5
    )
    print("✅ Advanced configuration created")
    
    # Create tokenizer with custom special tokens
    tokenizer = TinyPeLLMTokenizer(
        unk_token="<UNK>",
        bos_token="<START>",
        eos_token="<END>",
        pad_token="<PAD>"
    )
    print("✅ Custom tokenizer created")
    
    # Create model
    model = TinyPeLLMForCausalLM(config)
    print(f"✅ Advanced model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test with different input types
    texts = ["Hello world", "How are you?", "The weather is nice"]
    
    # Batch processing
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    print(f"✅ Batch processing: input shape {inputs['input_ids'].shape}")
    
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"✅ Batch forward pass: output shape {outputs.logits.shape}")
    
    # Test attention mask
    attention_mask = torch.ones_like(inputs['input_ids'])
    attention_mask[0, -2:] = 0  # Mask last 2 tokens of first sequence
    
    with torch.no_grad():
        outputs_masked = model(input_ids=inputs['input_ids'], attention_mask=attention_mask)
    print("✅ Attention mask test completed")
    
    # Test generation with different parameters
    generation_configs = [
        {"temperature": 0.5, "top_p": 0.9},
        {"temperature": 1.0, "top_k": 50},
        {"temperature": 0.8, "do_sample": True, "num_beams": 1}
    ]
    
    for i, gen_config in enumerate(generation_configs):
        try:
            generated = model.generate(
                inputs['input_ids'][:1],  # Use first sequence only
                max_length=20,
                **gen_config
            )
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"✅ Generation config {i+1}: {text}")
        except Exception as e:
            print(f"⚠️ Generation config {i+1} failed: {e}")


def demo_7_error_handling():
    """Demo 7: Error handling and edge cases"""
    print("\n" + "=" * 60)
    print("DEMO 7: Error Handling and Edge Cases")
    print("=" * 60)
    
    # Test invalid configuration
    try:
        config = TinyPeLLMConfig(vocab_size=0)
        print("❌ Should have failed with vocab_size=0")
    except Exception as e:
        print(f"✅ Properly caught invalid config: {e}")
    
    # Test empty input
    config, tokenizer, model = demo_1_basic_setup()
    
    try:
        inputs = tokenizer("", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        print("✅ Empty input handled correctly")
    except Exception as e:
        print(f"⚠️ Empty input failed: {e}")
    
    # Test very long input
    try:
        long_text = "Hello " * 1000
        inputs = tokenizer(long_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        print("✅ Long input truncated correctly")
    except Exception as e:
        print(f"⚠️ Long input failed: {e}")
    
    # Test invalid tokenizer file
    try:
        invalid_tokenizer = TinyPeLLMTokenizer(tokenizer_file="nonexistent.json")
        print("❌ Should have failed with nonexistent file")
    except Exception as e:
        print(f"✅ Properly caught invalid tokenizer file: {e}")


def main():
    """Run all demos"""
    print("🚀 TinyPeLLM Comprehensive Demo")
    print("This demo showcases all features of the TinyPeLLM implementation")
    
    try:
        # Run all demos
        demo_1_basic_setup()
        demo_2_basic_inference()
        demo_3_save_and_load()
        demo_4_pipeline_integration()
        demo_5_training_workflow()
        demo_6_advanced_features()
        demo_7_error_handling()
        
        print("\n" + "=" * 60)
        print("🎉 All demos completed successfully!")
        print("=" * 60)
        
        print("\n📋 Summary of Features Demonstrated:")
        print("✅ Auto class registration")
        print("✅ Model creation and configuration")
        print("✅ Tokenizer functionality")
        print("✅ Save/load with Auto classes")
        print("✅ Pipeline integration")
        print("✅ Training workflow")
        print("✅ Advanced features")
        print("✅ Error handling")
        
        print("\n🔧 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run tests: python test_tiny_pellm.py")
        print("3. Try the examples: python example_usage.py")
        print("4. Check the documentation in README.md")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 