import torch
import torch.nn as nn
import math

# --- Rotary Positional Embedding (RoPE) ---
def apply_rope(x):
    # x: (B, T, H, D)
    B, T, H, D = x.size()
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]

    theta = 1.0 / (10000 ** (torch.arange(0, half, 2).float() / half)).to(x.device)  # (half//2,)
    pos = torch.arange(T, dtype=torch.float32, device=x.device)  # (T,)
    freq = torch.einsum("i,j->ij", pos, theta)  # (T, half//2)

    sin = freq.sin()[None, :, None, :].repeat(B, 1, H, 1)  # (B, T, H, half//2)
    cos = freq.cos()[None, :, None, :].repeat(B, 1, H, 1)  # (B, T, H, half//2)

    x1_even, x1_odd = x1[..., ::2], x1[..., 1::2]
    x2_even, x2_odd = x2[..., ::2], x2[..., 1::2]

    x1_rot = torch.cat([x1_even * cos - x1_odd * sin, x1_even * sin + x1_odd * cos], dim=-1)
    x2_rot = torch.cat([x2_even * cos - x2_odd * sin, x2_even * sin + x2_odd * cos], dim=-1)

    return torch.cat([x1_rot, x2_rot], dim=-1)


# --- Multi-Query Attention (MQA) ---
class MultiQueryAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, self.head_dim)  # Shared key
        self.v_proj = nn.Linear(dim, self.head_dim)  # Shared value
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.size()
        H, D = self.n_heads, self.head_dim

        # Queries per head
        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)

        # Shared K and V (Multi-Query: 1 key and 1 value per token position)
        k = self.k_proj(x)  # (B, T, D)
        v = self.v_proj(x)  # (B, T, D)

        # Apply RoPE
        q = apply_rope(q)         # (B, H, T, D)
        k = apply_rope(k.unsqueeze(1))  # (B, 1, T, D)
        v = v.unsqueeze(1)        # (B, 1, T, D)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # (B, H, T, T)

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)  # (B, H, T, T)

        # Apply attention to value
        v = v.expand(-1, H, -1, -1)  # (B, H, T, D)
        attn_out = torch.matmul(attn_probs, v)  # (B, H, T, D)

        # Combine heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        return self.out_proj(attn_out)


# --- Feedforward Layer ---
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

# --- Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiQueryAttention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

# --- TinyPeLLM ---
class TinyPeLLM(nn.Module):
    def __init__(self, vocab_size, dim=128, n_layers=2, n_heads=4, hidden_dim=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, hidden_dim, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x):
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)
