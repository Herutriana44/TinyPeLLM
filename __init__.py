"""
TinyPeLLM - A lightweight language model with Hugging Face Transformers integration
"""

__version__ = "0.1.0"
__author__ = "TinyPeLLM Team"
__email__ = "contact@tinypellm.com"

# Import main classes
from .TinyPeLLMModel import TinyPeLLMConfig, TinyPeLLMForCausalLM, TinyPeLLMModel
from .TinyPeLLMTokenizer import TinyPeLLMTokenizer
from .TrainerTinyPeLLMPipeline import (
    register_tiny_pellm,
    TinyPeLLMTrainer,
    TinyPeLLMPipeline,
    create_tiny_pellm_pipeline
)

# Auto-register the model
try:
    register_tiny_pellm()
except Exception as e:
    import warnings
    warnings.warn(f"Failed to auto-register TinyPeLLM: {e}")

__all__ = [
    "TinyPeLLMConfig",
    "TinyPeLLMForCausalLM", 
    "TinyPeLLMModel",
    "TinyPeLLMTokenizer",
    "register_tiny_pellm",
    "TinyPeLLMTrainer",
    "TinyPeLLMPipeline",
    "create_tiny_pellm_pipeline",
] 