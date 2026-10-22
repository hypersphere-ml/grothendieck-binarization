"""GPT-2 small (124.4M parameters), as in nanoGPT: learned positions, pre-norm blocks, output layer tied to the token
embedding."""
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

N_LAYER, N_HEAD, N_EMBD, VOCAB, BLOCK = 12, 12, 768, 50257, 1024


class Attention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = (t.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2) for t in self.c_attn(x).split(C, dim=2))
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        return self.resid_dropout(self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C)))


class MLP(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBD)
        self.attn = Attention(dropout)
        self.ln_2 = nn.LayerNorm(N_EMBD)
        self.mlp = MLP(dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(VOCAB, N_EMBD),
            wpe=nn.Embedding(BLOCK, N_EMBD),
            drop=nn.Dropout(dropout),
            h=nn.ModuleList([Block(dropout) for _ in range(N_LAYER)]),
            ln_f=nn.LayerNorm(N_EMBD),
        ))
        self.lm_head = nn.Linear(N_EMBD, VOCAB, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        for name, p in self.named_parameters():
            if name.endswith("bias"):
                nn.init.zeros_(p)
            elif p.dim() == 2:
                std = 0.02 / math.sqrt(2 * N_LAYER) if name.endswith("c_proj.weight") else 0.02
                nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, idx, targets):
        """The mean next-token cross-entropy."""
        x = self.transformer.wte(idx) + self.transformer.wpe(torch.arange(idx.shape[1], device=idx.device))
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))


def body(model):
    """The 48 linear layers of the transformer blocks, as (name, layer), in block order."""
    pattern = r"transformer\.h\.\d+\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)"
    return [(name, m) for name, m in model.named_modules() if re.fullmatch(pattern, name)]


def load(path, device):
    model = GPT().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model
