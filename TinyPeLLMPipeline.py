"""
TinyPeLLM Pipeline and Trainer Implementation for Hugging Face Transformers
Provides registration with Auto classes and training utilities
"""

import os
import json
import torch
from typing import Optional, Dict, Any, List, Union
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    Trainer,
    TrainingArguments,
    pipeline,
    DataCollatorForLanguageModeling,
    TextDataset,
    LineByLineTextDataset
)

from transformers.pipelines import PIPELINE_REGISTRY
from transformers.utils import logging
from datasets import Dataset
import numpy as np

# Import our custom classes
from TinyPeLLMConfig import TinyPeLLMConfig
from TinyPeLLMModel import TinyPeLLMForCausalLM
from TinyPeLLMTokenizer import TinyPeLLMTokenizer

logger = logging.get_logger(__name__)


def register_tiny_pellm():
    """
    Register TinyPeLLM model and tokenizer with Auto classes
    """
    try:
        # Register config
        AutoConfig.register("tiny_pellm", TinyPeLLMConfig)
        
        # Register tokenizer - FIXED: Use proper registration for fast tokenizer
        AutoTokenizer.register(TinyPeLLMConfig, TinyPeLLMTokenizer)
        
        # Register model
        AutoModelForCausalLM.register(TinyPeLLMConfig, TinyPeLLMForCausalLM)
        
        logger.info("TinyPeLLM successfully registered with Auto classes")
        
    except Exception as e:
        logger.error(f"Failed to register TinyPeLLM: {e}")
        raise


class TinyPeLLMDataCollator(DataCollatorForLanguageModeling):
    """
    Custom data collator for TinyPeLLM training
    """
    def __init__(self, tokenizer, mlm=False, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=mlm, **kwargs)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Call parent method
        batch = super().__call__(features)
        
        # Ensure proper tensor types
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].long()
        
        return batch


class TinyPeLLMTrainer:
    """
    Custom trainer class for TinyPeLLM fine-tuning
    """
    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        config: Optional[TinyPeLLMConfig] = None,
        tokenizer: Optional[TinyPeLLMTokenizer] = None,
        **kwargs
    ):
        self.model_name_or_path = model_name_or_path
        self.config = config
        self.tokenizer = tokenizer
        self.trainer = None
        
        # Register the model if not already registered
        try:
            AutoConfig.get_config_dict("tiny_pellm")
        except:
            register_tiny_pellm()
    
    def prepare_model_and_tokenizer(
        self,
        vocab_size: int = 3000,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        **kwargs
    ):
        """
        Prepare model and tokenizer for training
        """
        # Create config if not provided
        if self.config is None:
            self.config = TinyPeLLMConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                **kwargs
            )
        
        # Create tokenizer if not provided
        if self.tokenizer is None:
            self.tokenizer = TinyPeLLMTokenizer()
        
        # Create model
        self.model = TinyPeLLMForCausalLM(self.config)
        
        return self.model, self.tokenizer
    
    def prepare_dataset(
        self,
        data_path: str,
        block_size: int = 128,
        overwrite_cache: bool = False,
        **kwargs
    ) -> Dataset:
        """
        Prepare dataset for training
        """
        if os.path.isfile(data_path):
            # Single file
            extension = data_path.split(".")[-1]
            if extension == "txt":
                dataset = LineByLineTextDataset(
                    tokenizer=self.tokenizer,
                    file_path=data_path,
                    block_size=block_size,
                    overwrite_cache=overwrite_cache,
                )
            else:
                raise ValueError(f"Unsupported file extension: {extension}")
        else:
            # Directory
            dataset = TextDataset(
                tokenizer=self.tokenizer,
                file_path=data_path,
                block_size=block_size,
                overwrite_cache=overwrite_cache,
            )
        
        return dataset
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "./tiny_pellm_output",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        logging_steps: int = 100,
        save_steps: int = 1000,
        eval_steps: Optional[int] = None,
        save_total_limit: Optional[int] = None,
        **kwargs
    ):
        """
        Train the model
        """
        # Prepare model and tokenizer if not already done
        if not hasattr(self, 'model'):
            self.prepare_model_and_tokenizer()
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            save_total_limit=save_total_limit,
            prediction_loss_only=True,
            remove_unused_columns=False,
            **kwargs
        )
        
        # Create data collator
        data_collator = TinyPeLLMDataCollator(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        train_result = self.trainer.train()
        
        # Save the model
        self.save_model(output_dir)
        
        return train_result
    
    def save_model(self, output_dir: str):
        """
        Save the trained model
        """
        if self.trainer is not None:
            self.trainer.save_model(output_dir)
        
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        if self.config is not None:
            self.config.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, model_path: str):
        """
        Load a trained model
        """
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
        
        return self.model, self.tokenizer


def create_tiny_pellm_pipeline(
    model_name_or_path: str,
    device: Optional[int] = None,
    **kwargs
):
    """
    Create a TinyPeLLM pipeline for text generation
    """
    # Register the model if not already registered
    try:
        AutoConfig.get_config_dict("tiny_pellm")
    except:
        register_tiny_pellm()
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model_name_or_path,
        device=device,
        **kwargs
    )
    
    return pipe


class TinyPeLLMPipeline:
    """
    Custom pipeline class for TinyPeLLM
    """
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[int] = None,
        **kwargs
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.pipe = create_tiny_pellm_pipeline(model_name_or_path, device, **kwargs)
    
    def generate(
        self,
        text: str,
        max_length: int = 100,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate text using the pipeline
        """
        outputs = self.pipe(
            text,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            **kwargs
        )
        
        # Extract generated text
        generated_texts = []
        for output in outputs:
            if isinstance(output, list):
                for seq in output:
                    generated_texts.append(seq['generated_text'])
            else:
                generated_texts.append(output['generated_text'])
        
        return generated_texts


# Register custom pipeline (fixed version)
def tiny_pellm_pipeline(
    model,
    tokenizer,
    **kwargs
):
    """
    Custom pipeline for TinyPeLLM text generation
    """
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **kwargs
    )

# Register the pipeline function
try:
    PIPELINE_REGISTRY.register_pipeline("text-generation-tiny-pellm", tiny_pellm_pipeline)
except Exception as e:
    logger.warning(f"Could not register custom pipeline: {e}")


def main():
    """
    Example usage of TinyPeLLM training and inference
    """
    # Register the model
    register_tiny_pellm()
    
    # Initialize trainer
    trainer = TinyPeLLMTrainer()
    
    # Prepare model and tokenizer
    model, tokenizer = trainer.prepare_model_and_tokenizer(
        vocab_size=3000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4
    )
    
    # Prepare dataset
    dataset = trainer.prepare_dataset(
        data_path="tinypellm_corpus.txt",
        block_size=128
    )
    
    # Train the model
    train_result = trainer.train(
        train_dataset=dataset,
        output_dir="./tiny_pellm_trained",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=5e-5
    )
    
    # Create pipeline for inference
    pipe = TinyPeLLMPipeline("./tiny_pellm_trained")
    
    # Generate text
    generated_texts = pipe.generate(
        text="Hello, how are you?",
        max_length=50,
        temperature=0.8
    )
    
    print("Generated text:", generated_texts)


if __name__ == "__main__":
    main()
