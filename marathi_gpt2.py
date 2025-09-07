
import torch
import torch.nn as nn
import math

class GPT2Config:
    def __init__(self, vocab_size=32000, n_positions=1024, n_embd=768, n_layer=12, n_head=12, dropout=0.1):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout

class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head, dropout=config.dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x, attn_mask=None):
        # Pre-norm
        x_norm = self.ln_1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids):
        bsz, seq_len = input_ids.size()
        pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        pos = pos.unsqueeze(0).expand(bsz, seq_len)
        x = self.wte(input_ids) + self.wpe(pos)
        x = self.drop(x)
        # Causal mask: [seq_len, seq_len], True means masked
        mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        for block in self.h:
            x = block(x, attn_mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# Example usage:
# config = GPT2Config()
# model = GPT2Model(config)
# input_ids = torch.randint(0, config.vocab_size, (2, 32))
# logits = model(input_ids)
