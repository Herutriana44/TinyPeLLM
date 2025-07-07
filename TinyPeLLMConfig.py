"""
TinyPeLLM Configuration Class
Separate file to avoid circular imports
"""

from transformers import PretrainedConfig


class TinyPeLLMConfig(PretrainedConfig):
    """
    Configuration class for TinyPeLLM model
    """
    model_type = "tiny_pellm"
    
    def __init__(
        self,
        vocab_size: int = 3000,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 4,
        intermediate_size: int = 256,
        hidden_act: str = "gelu",
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        tie_word_embeddings: bool = True,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache 