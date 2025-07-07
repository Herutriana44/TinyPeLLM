"""
Example Usage of TinyPeLLM with Hugging Face Transformers
Demonstrates registration with Auto classes and pipeline usage
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, PreTrainedTokenizerFast
from TinyPeLLMModel import TinyPeLLMConfig, TinyPeLLMForCausalLM
from TinyPeLLMTokenizer import TinyPeLLMTokenizer
from TinyPeLLMPipeline import register_tiny_pellm, TinyPeLLMTrainer, TinyPeLLMPipeline


def example_1_basic_registration():
    """
    Example 1: Basic registration and usage with Auto classes
    """
    print("=== Example 1: Basic Registration ===")
    
    # Register the model with Auto classes
    register_tiny_pellm()
    
    # Create config
    config = TinyPeLLMConfig(
        vocab_size=3000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4
    )
    
    # Create tokenizer
    tokenizer = TinyPeLLMTokenizer()
    
    # Create model
    model = TinyPeLLMForCausalLM(config)

    model.lm_head.weight = model.model.embed_tokens.weight
    
    # Save model and tokenizer
    model_path = "./tiny_pellm_model"
    model.save_pretrained(model_path, safe_serialization=False)
    tokenizer.save_pretrained(model_path)
    
    print(f"Model saved to {model_path}")
    
    # Load using Auto classes
    loaded_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    loaded_model = AutoModelForCausalLM.from_pretrained(model_path)
    
    print("Successfully loaded model and tokenizer using Auto classes!")
    return loaded_model, loaded_tokenizer


def example_2_pipeline_usage():
    """
    Example 2: Using the model with Hugging Face pipeline
    """
    print("\n=== Example 2: Pipeline Usage ===")
    
    # Register the model
    register_tiny_pellm()
    
    # Create and save a simple model
    config = TinyPeLLMConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2
    )
    
    tokenizer = TinyPeLLMTokenizer()
    model = TinyPeLLMForCausalLM(config)
    
    model_path = "./tiny_pellm_pipeline_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model_path,
        tokenizer=model_path,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Generate text
    text = "Hello, how are you?"
    outputs = pipe(
        text,
        max_length=50,
        num_return_sequences=1,
        temperature=0.8,
        do_sample=True
    )
    
    print(f"Input: {text}")
    print(f"Generated: {outputs[0]['generated_text']}")
    
    return pipe


def example_3_custom_pipeline():
    """
    Example 3: Using custom TinyPeLLM pipeline
    """
    print("\n=== Example 3: Custom Pipeline ===")
    
    # Register the model
    register_tiny_pellm()
    
    # Create and save a simple model
    config = TinyPeLLMConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2
    )
    
    tokenizer = TinyPeLLMTokenizer()
    model = TinyPeLLMForCausalLM(config)
    
    model_path = "./tiny_pellm_custom_pipeline"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Use custom pipeline
    custom_pipe = TinyPeLLMPipeline(model_path)
    
    # Generate text
    generated_texts = custom_pipe.generate(
        text="The weather is",
        max_length=30,
        temperature=0.7,
        num_return_sequences=2
    )
    
    print(f"Input: The weather is")
    for i, text in enumerate(generated_texts):
        print(f"Generated {i+1}: {text}")


def example_4_training():
    """
    Example 4: Training the model
    """
    print("\n=== Example 4: Training ===")
    
    # Register the model
    register_tiny_pellm()
    
    # Initialize trainer
    trainer = TinyPeLLMTrainer()
    
    # Prepare model and tokenizer
    model, tokenizer = trainer.prepare_model_and_tokenizer(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2
    )
    
    # Create a simple training dataset
    training_texts = [
        "Hello world!",
        "How are you today?",
        "The weather is nice.",
        "I love programming.",
        "Machine learning is fascinating."
    ]
    
    # Save training data
    with open("simple_training_data.txt", "w", encoding="utf-8") as f:
        for text in training_texts:
            f.write(text + "\n")
    
    # Prepare dataset
    dataset = trainer.prepare_dataset(
        data_path="simple_training_data.txt",
        block_size=32
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Train the model (with minimal epochs for demo)
    try:
        train_result = trainer.train(
            train_dataset=dataset,
            output_dir="./tiny_pellm_trained",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            learning_rate=1e-3,
            logging_steps=1,
            save_steps=10
        )
        
        print("Training completed successfully!")
        
        # Test the trained model
        trained_pipe = TinyPeLLMPipeline("./tiny_pellm_trained")
        generated = trained_pipe.generate("Hello", max_length=20)
        print(f"Trained model output: {generated[0]}")
        
    except Exception as e:
        print(f"Training failed (this is expected for demo): {e}")


def example_5_advanced_usage():
    """
    Example 5: Advanced usage with custom configuration
    """
    print("\n=== Example 5: Advanced Usage ===")
    
    # Register the model
    register_tiny_pellm()
    
    # Create custom config
    config = TinyPeLLMConfig(
        vocab_size=5000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=512,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5
    )
    
    # Create tokenizer with custom special tokens
    tokenizer = TinyPeLLMTokenizer(
        unk_token="<UNK>",
        bos_token="<START>",
        eos_token="<END>",
        pad_token="<PAD>"
    )
    
    # Create model
    model = TinyPeLLMForCausalLM(config)
    
    # Test forward pass
    input_text = "Hello world"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Output shape: {outputs.logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save and load
    model_path = "./tiny_pellm_advanced"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Load using Auto classes
    loaded_model = AutoModelForCausalLM.from_pretrained(model_path)
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Advanced model successfully saved and loaded!")


def main():
    """
    Run all examples
    """
    print("TinyPeLLM Hugging Face Integration Examples")
    print("=" * 50)
    
    try:
        # Example 1: Basic registration
        example_1_basic_registration()
        
        # Example 2: Pipeline usage
        example_2_pipeline_usage()
        
        # Example 3: Custom pipeline
        example_3_custom_pipeline()
        
        # Example 4: Training
        example_4_training()
        
        # Example 5: Advanced usage
        example_5_advanced_usage()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nUsage Summary:")
        print("1. Register model: register_tiny_pellm()")
        print("2. Use Auto classes: AutoTokenizer.from_pretrained(), AutoModelForCausalLM.from_pretrained()")
        print("3. Use pipeline: pipeline('text-generation', model=model_path)")
        print("4. Use custom pipeline: TinyPeLLMPipeline(model_path)")
        print("5. Train model: TinyPeLLMTrainer().train()")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 