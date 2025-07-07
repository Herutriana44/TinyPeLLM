"""
TinyPeLLM Model Implementation
Clean version without circular imports
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union, List
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

# Import config from separate file
from TinyPeLLMConfig import TinyPeLLMConfig

logger = logging.get_logger(__name__)


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Rotary Positional Embedding (RoPE) to input tensor
    """
    # x: (B, T, H, D)
    B, T, H, D = x.size()
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]

    theta = 1.0 / (10000 ** (torch.arange(0, half, 2).float() / half)).to(x.device)
    pos = torch.arange(T, dtype=torch.float32, device=x.device)
    freq = torch.einsum("i,j->ij", pos, theta)

    sin = freq.sin()[None, :, None, :].repeat(B, 1, H, 1)
    cos = freq.cos()[None, :, None, :].repeat(B, 1, H, 1)

    x1_even, x1_odd = x1[..., ::2], x1[..., 1::2]
    x2_even, x2_odd = x2[..., ::2], x2[..., 1::2]

    x1_rot = torch.cat([x1_even * cos - x1_odd * sin, x1_even * sin + x1_odd * cos], dim=-1)
    x2_rot = torch.cat([x2_even * cos - x2_odd * sin, x2_even * sin + x2_odd * cos], dim=-1)

    return torch.cat([x1_rot, x2_rot], dim=-1)


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention implementation
    """
    def __init__(self, config: TinyPeLLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.size()
        H, D = self.num_attention_heads, self.head_dim

        # Queries per head
        q = self.q_proj(hidden_states).view(batch_size, seq_length, H, D).transpose(1, 2)

        # Shared K and V (Multi-Query: 1 key and 1 value per token position)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Apply RoPE
        q = apply_rope(q)
        k = apply_rope(k.unsqueeze(1))
        v = v.unsqueeze(1)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Causal mask
        mask = torch.tril(torch.ones(seq_length, seq_length, device=hidden_states.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)

        # Apply attention to value
        v = v.expand(-1, H, -1, -1)
        attn_out = torch.matmul(attn_probs, v)

        # Combine heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)

        return self.out_proj(attn_out)


class FeedForward(nn.Module):
    """
    Feedforward layer implementation
    """
    def __init__(self, config: TinyPeLLMConfig):
        super().__init__()
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        intermediate = self.activation(self.intermediate(hidden_states))
        return self.output(intermediate)


class TransformerBlock(nn.Module):
    """
    Transformer block implementation
    """
    def __init__(self, config: TinyPeLLMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = MultiQueryAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ff = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.ff(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TinyPeLLMModel(PreTrainedModel):
    """
    TinyPeLLM model implementation compatible with Hugging Face Transformers
    """
    config_class = TinyPeLLMConfig
    base_model_prefix = "tiny_pellm"
    supports_gradient_checkpointing = True

    def __init__(self, config: TinyPeLLMConfig):
        super().__init__(config)
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.post_init()

    def post_init(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Retrieve input_ids
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        hidden_states = inputs_embeds

        # Apply transformer layers
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return CausalLMOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> dict:
        # If past_key_values is provided, we only need the last token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "past_key_values": past_key_values,
        }

    def get_output_embeddings(self) -> nn.Module:
        return None

    def set_output_embeddings(self, new_embeddings: nn.Module):
        pass


class TinyPeLLMForCausalLM(PreTrainedModel):
    """
    TinyPeLLM model with language modeling head for causal language modeling
    """
    config_class = TinyPeLLMConfig
    base_model_prefix = "tiny_pellm"
    supports_gradient_checkpointing = True

    def __init__(self, config: TinyPeLLMConfig):
        super().__init__(config)
        self.config = config
        self.model = TinyPeLLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> dict:
        return self.model.prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs
        )

    def _reorder_cache(self, past_key_values: List[torch.FloatTensor], beam_idx: int) -> List[torch.FloatTensor]:
        return self.model._reorder_cache(past_key_values, beam_idx) 