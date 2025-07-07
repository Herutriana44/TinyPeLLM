import torch
import sentencepiece as spm

class TinyPeLLMTokenizer:
    def __init__(self, model_path="tinypellm.model", bos_token_id=2, eos_token_id=3, device="cpu"):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.device = device

    def encode(self, text, add_special_tokens=True):
        ids = self.sp.encode(text, out_type=int)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return torch.tensor(ids, dtype=torch.long).to(self.device)

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.sp.decode(token_ids)

    def batch_encode(self, texts, add_special_tokens=True, padding=False, max_length=None):
        all_ids = []
        for text in texts:
            ids = self.sp.encode(text, out_type=int)
            if add_special_tokens:
                ids = [self.bos_token_id] + ids + [self.eos_token_id]
            all_ids.append(ids)

        if padding:
            max_len = max_length or max(len(ids) for ids in all_ids)
            all_ids = [ids + [0] * (max_len - len(ids)) for ids in all_ids]

        return torch.tensor(all_ids, dtype=torch.long).to(self.device)

    def vocab_size(self):
        return self.sp.get_piece_size()

    def token_to_id(self, token):
        return self.sp.piece_to_id(token)

    def id_to_token(self, idx):
        return self.sp.id_to_piece(idx)

    def detokenize(self, token_ids):
        return self.decode(token_ids)
