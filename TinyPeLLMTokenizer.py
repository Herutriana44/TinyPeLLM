"""
TinyPeLLM Tokenizer Implementation for Hugging Face Transformers
A custom tokenizer that can be registered with AutoTokenizer
"""

import os
import json
from typing import List, Optional, Union, Dict, Any, Tuple
from transformers import PreTrainedTokenizerFast
from transformers.utils import logging
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

logger = logging.get_logger(__name__)


class TinyPeLLMTokenizer(PreTrainedTokenizerFast):
    """
    TinyPeLLM Tokenizer implementation compatible with Hugging Face Transformers
    """
    vocab_files_names = {
        "vocab_file": "vocab.json",
        "tokenizer_file": "tokenizer.json",
        "merges_file": "merges.txt"
    }
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        tokenizer_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        cls_token: str = "<cls>",
        mask_token: str = "<mask>",
        sep_token: str = "<sep>",
        **kwargs
    ):
        # Initialize the tokenizer
        if tokenizer_file is not None:
            tokenizer = HFTokenizer.from_file(tokenizer_file)
        elif vocab_file is not None:
            # Create tokenizer from vocab file
            tokenizer = self._create_tokenizer_from_vocab(vocab_file, merges_file)
        else:
            # Create a new tokenizer
            tokenizer = self._create_new_tokenizer()
        
        # Set special tokens
        special_tokens = {
            "unk_token": unk_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "pad_token": pad_token,
            "cls_token": cls_token,
            "mask_token": mask_token,
            "sep_token": sep_token,
        }
        
        super().__init__(
            tokenizer_object=tokenizer,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            sep_token=sep_token,
            **kwargs
        )
        
        # Set post processor
        self._tokenizer.post_processor = TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            pair=f"{self.bos_token} $A {self.sep_token} $B:1 {self.eos_token}:1",
            special_tokens=[
                (self.bos_token, self.bos_token_id),
                (self.eos_token, self.eos_token_id),
                (self.sep_token, self.sep_token_id),
            ],
        )

    def _create_tokenizer_from_vocab(self, vocab_file: str = vocab_files_names["vocab_file"], merges_file: Optional[str] = vocab_files_names["merges_file"]) -> HFTokenizer:
        """
        Create a tokenizer from existing vocab and merges files
        """
        # Load vocabulary
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        # Create BPE model
        if merges_file and os.path.exists(merges_file):
            with open(merges_file, 'r', encoding='utf-8') as f:
                merges = [line.strip() for line in f if line.strip()]
            model = BPE(vocab, merges)
        else:
            model = BPE(vocab)
        
        tokenizer = HFTokenizer(model)
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        
        return tokenizer

    def _create_new_tokenizer(self) -> HFTokenizer:
        """
        Create a new tokenizer from scratch
        """
        tokenizer = HFTokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        
        return tokenizer

    def train_from_files(
        self,
        files: List[str],
        vocab_size: int = 3000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Train the tokenizer from files
        """
        if special_tokens is None:
            special_tokens = [
                self.unk_token,
                self.bos_token,
                self.eos_token,
                self.pad_token,
                self.cls_token,
                self.mask_token,
                self.sep_token,
            ]
        
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            **kwargs
        )
        
        self._tokenizer.train(files, trainer)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs
    ) -> Tuple[str, ...]:
        """
        Save the tokenizer to a directory
        """
        save_directory = str(save_directory)
        os.makedirs(save_directory, exist_ok=True)
        
        # Save tokenizer files
        tokenizer_file = os.path.join(save_directory, "tokenizer.json")
        self._tokenizer.save(tokenizer_file)
        
        # Save vocab file
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self._tokenizer.get_vocab(), f, ensure_ascii=False, indent=2)
        
        # Save merges file if available
        if hasattr(self._tokenizer.model, 'merges'):
            merges_file = os.path.join(save_directory, "merges.txt")
            with open(merges_file, 'w', encoding='utf-8') as f:
                for merge in self._tokenizer.model.merges:
                    f.write(f"{merge}\n")
        
        # Save config
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        config = {
            "tokenizer_class": self.__class__.__name__,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "cls_token": self.cls_token,
            "mask_token": self.mask_token,
            "sep_token": self.sep_token,
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        return (tokenizer_file, vocab_file)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *model_args,
        **kwargs
    ) -> "TinyPeLLMTokenizer":
        """
        Load a tokenizer from a pretrained model
        """
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def encode_plus(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, type]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Encode text with additional features
        """
        return super().encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs
        )

    def decode(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs
    ) -> str:
        """
        Decode token IDs back to text
        """
        return super().decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs
    ) -> List[str]:
        """
        Decode a batch of token IDs
        """
        return super().batch_decode(
            sequences=sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs
        )

    def get_vocab(self) -> Dict[str, int]:
        """
        Get the vocabulary
        """
        return self._tokenizer.get_vocab()

    def vocab_size(self) -> int:
        """
        Get the vocabulary size
        """
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        """
        Convert a token to its ID
        """
        return self._tokenizer.token_to_id(token)

    def id_to_token(self, id: int) -> str:
        """
        Convert an ID to its token
        """
        return self._tokenizer.id_to_token(id)

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Convert tokens to IDs
        """
        if isinstance(tokens, str):
            return self.token_to_id(tokens)
        return [self.token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]], skip_special_tokens: bool = False) -> Union[str, List[str]]:
        """
        Convert IDs to tokens
        """
        if isinstance(ids, int):
            token = self.id_to_token(ids)
            if skip_special_tokens and token in self.all_special_tokens:
                return ""
            return token
        
        tokens = [self.id_to_token(id) for id in ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.all_special_tokens]
        return tokens
