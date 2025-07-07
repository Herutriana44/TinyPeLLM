# TinyPeLLM - Hugging Face Transformers Integration

A lightweight language model implementation that can be registered with Hugging Face's Auto classes (`AutoTokenizer`, `AutoModelForCausalLM`, `AutoPipeline`) for seamless integration with the transformers ecosystem.

## Features

- ✅ **Auto Registration**: Compatible with `AutoTokenizer`, `AutoModelForCausalLM`, and `AutoPipeline`
- ✅ **Custom Architecture**: Multi-Query Attention with Rotary Positional Embeddings (RoPE)
- ✅ **Training Support**: Full integration with Hugging Face Trainer
- ✅ **Pipeline Integration**: Works with transformers pipeline for text generation
- ✅ **Save/Load**: Complete `save_pretrained()` and `from_pretrained()` support
- ✅ **Type Hints**: Full type annotations for better development experience
- ✅ **Error Handling**: Robust error handling and logging

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Usage with Auto Classes

```python
from TrainerTinyPeLLMPipeline import register_tiny_pellm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Register the model
register_tiny_pellm()

# Create and save model
from TinyPeLLMModel import TinyPeLLMConfig, TinyPeLLMForCausalLM
from TinyPeLLMTokenizer import TinyPeLLMTokenizer

config = TinyPeLLMConfig(vocab_size=3000, hidden_size=128)
tokenizer = TinyPeLLMTokenizer()
model = TinyPeLLMForCausalLM(config)

# Save model
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")

# Load using Auto classes
tokenizer = AutoTokenizer.from_pretrained("./my_model")
model = AutoModelForCausalLM.from_pretrained("./my_model")

# Use with pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
result = pipe("Hello, how are you?", max_length=50)
print(result[0]['generated_text'])
```

### 2. Training

```python
from TrainerTinyPeLLMPipeline import TinyPeLLMTrainer

# Initialize trainer
trainer = TinyPeLLMTrainer()

# Prepare model and dataset
model, tokenizer = trainer.prepare_model_and_tokenizer(
    vocab_size=3000,
    hidden_size=128,
    num_hidden_layers=2
)

dataset = trainer.prepare_dataset("your_data.txt", block_size=128)

# Train
trainer.train(
    train_dataset=dataset,
    output_dir="./trained_model",
    num_train_epochs=3,
    learning_rate=5e-5
)
```

### 3. Custom Pipeline

```python
from TrainerTinyPeLLMPipeline import TinyPeLLMPipeline

# Create custom pipeline
pipe = TinyPeLLMPipeline("./my_model")

# Generate text
texts = pipe.generate(
    text="The weather is",
    max_length=100,
    temperature=0.8,
    num_return_sequences=3
)

for text in texts:
    print(text)
```

## Architecture

### Model Components

1. **Multi-Query Attention (MQA)**: Efficient attention mechanism with shared key/value projections
2. **Rotary Positional Embeddings (RoPE)**: Relative positional encoding for better sequence modeling
3. **Transformer Blocks**: Standard transformer architecture with layer normalization and residual connections
4. **Language Modeling Head**: Linear projection to vocabulary size for next-token prediction

### Configuration

```python
TinyPeLLMConfig(
    vocab_size=3000,              # Vocabulary size
    hidden_size=128,              # Hidden dimension
    num_hidden_layers=2,          # Number of transformer layers
    num_attention_heads=4,        # Number of attention heads
    intermediate_size=256,        # Feedforward hidden size
    max_position_embeddings=2048, # Maximum sequence length
    initializer_range=0.02,       # Weight initialization range
    layer_norm_eps=1e-5,          # Layer normalization epsilon
    pad_token_id=0,               # Padding token ID
    bos_token_id=2,               # Beginning of sequence token ID
    eos_token_id=3,               # End of sequence token ID
)
```

## File Structure

```
TinyPeLLM/
├── TinyPeLLMModel.py              # Model implementation
├── TinyPeLLMTokenizer.py          # Tokenizer implementation
├── TrainerTinyPeLLMPipeline.py    # Training and pipeline utilities
├── requirements.txt               # Dependencies
├── example_usage.py              # Usage examples
├── README.md                     # This file
└── tinypellm/                    # Model files
    ├── tinypellm.model           # SentencePiece model
    ├── tinypellm.vocab           # Vocabulary file
    └── tinypellm_corpus.txt      # Training corpus
```

## API Reference

### TinyPeLLMConfig

Configuration class for the TinyPeLLM model.

```python
config = TinyPeLLMConfig(
    vocab_size=3000,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=4
)
```

### TinyPeLLMForCausalLM

Main model class for causal language modeling.

```python
model = TinyPeLLMForCausalLM(config)
outputs = model(input_ids, attention_mask=attention_mask)
```

### TinyPeLLMTokenizer

Custom tokenizer implementation.

```python
tokenizer = TinyPeLLMTokenizer()
encoded = tokenizer.encode("Hello world")
decoded = tokenizer.decode(encoded)
```

### TinyPeLLMTrainer

Training utility class.

```python
trainer = TinyPeLLMTrainer()
model, tokenizer = trainer.prepare_model_and_tokenizer()
trainer.train(train_dataset, output_dir="./output")
```

### TinyPeLLMPipeline

Custom pipeline for text generation.

```python
pipe = TinyPeLLMPipeline("./model_path")
texts = pipe.generate("Hello", max_length=50)
```

## Training

### Data Preparation

The model expects text data in the following format:
- Single text file with one sentence per line
- UTF-8 encoding
- No special preprocessing required

### Training Configuration

```python
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    warmup_steps=500,
    logging_steps=100,
    save_steps=1000,
    prediction_loss_only=True
)
```

### Training Process

1. **Register Model**: Call `register_tiny_pellm()` to register with Auto classes
2. **Prepare Data**: Use `trainer.prepare_dataset()` to create training dataset
3. **Train**: Call `trainer.train()` with your dataset
4. **Save**: Model is automatically saved to the specified output directory

## Inference

### Using Pipeline

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="./model_path")
result = pipe("Hello, how are you?", max_length=100)
```

### Using Model Directly

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./model_path")
model = AutoModelForCausalLM.from_pretrained("./model_path")

inputs = tokenizer("Hello", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Examples

Run the complete example suite:

```bash
python example_usage.py
```

This will demonstrate:
1. Basic registration and Auto class usage
2. Pipeline integration
3. Custom pipeline usage
4. Training workflow
5. Advanced configuration

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.21+
- Tokenizers 0.12+
- Datasets 2.0+
- Other dependencies listed in `requirements.txt`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Hugging Face Transformers library
- SentencePiece for tokenization
- PyTorch for deep learning framework 