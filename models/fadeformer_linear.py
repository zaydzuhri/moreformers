"""
GPT Model definition, based on the nanoGPT implementation of GPT-2, but simpler.
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

def calc_fade(n_input):
    fade_steps = [n_input]
    while n_input > 64:
        n_output = n_input//2
        fade_steps.append(n_output)
        n_input = n_output
    return fade_steps

# Layer normalization with optional bias
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

# Causal Self Attention Head without flash attention
class Head(nn.Module):
    def __init__(self, config, ctx_size):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.query = nn.Linear(config.n_embd, head_size)
        self.key = nn.Linear(config.n_embd, head_size)
        self.value = nn.Linear(config.n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(ctx_size, ctx_size)))
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        # get query and key projections
        q = self.query(x) # (B, T, C)
        k = self.key(x) # (B, T, C)
        # compute attention "affinities", scale, mask, and softmax
        att = q @ k.transpose(-2, -1) # (B, T, C) @ (B, C, T) -> (B, T, T)
        att = att * C ** (-0.5)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        # apply attention to value projection
        v = self.value(x) # (B, T, C)
        out = att @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

# Parallel attention heads
class MultiHeadAttention(nn.Module):
    def __init__(self, config, ctx_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(config, ctx_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias) # linear for dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # compute and concat heads in parallel
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        # project and dropout
        out = self.dropout(self.proj(out))
        return out

# Feedforward layer
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lin1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # linear, gelu, linear, dropout
        out = self.lin1(x)
        out = self.gelu(out)
        out = self.lin2(out)
        out = self.dropout(out)
        return out

# Fade block
class Fade(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        n_output = n_input//2
        self.out_sizes = [n_output//8, n_output//8, n_output//4, n_output//2]
        n_rest = n_input - n_output//2
        self.in_sizes = [n_rest//2, n_rest//4, n_rest//4, n_output//2]
        self.lin1 = nn.Linear(self.in_sizes[0], self.out_sizes[0], bias=False)
        self.lin2 = nn.Linear(self.in_sizes[1], self.out_sizes[1], bias=False)
        self.lin3 = nn.Linear(self.in_sizes[2], self.out_sizes[2], bias=False)

    def forward(self, x): # (B, T, C)
        # turn x to (B, C, T)
        x4 = x[:, -self.in_sizes[3]:]
        x = x.transpose(1, 2)
        x1 = self.lin1(x[:, :, :self.in_sizes[0]])
        x2 = self.lin2(x[:, :, self.in_sizes[0]:self.in_sizes[0]+self.in_sizes[1]])
        x3 = self.lin3(x[:, :, self.in_sizes[0]+self.in_sizes[1]:self.in_sizes[0]+self.in_sizes[1]+self.in_sizes[2]])
        # turn back to (B, T/2, C)
        x1 = x1.transpose(1, 2)
        x2 = x2.transpose(1, 2)
        x3 = x3.transpose(1, 2)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        return x

# Transformer block, attention for "communication", feedforward for "computation"
class FadingBlock(nn.Module):
    def __init__(self, config, n_time):
        super().__init__()
        self.csa = MultiHeadAttention(config, n_time)
        self.ff = FeedForward(config)
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.fade = Fade(n_time)
    
    def forward(self, x):
        # layer norm, attention, residual
        out = x + self.csa(self.ln1(x))
        # layer norm, feedforward, residual
        out = out + self.ff(self.ln2(out))
        # fade
        out = self.fade(out)
        # # fade and add half the time dimension of the previous output (residual)
        # out = out[:, out.shape[1]//2:] + self.fade(out)
        return out

# FadeFormer Model
class FadeFormerLinear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embd = nn.Embedding(config.ctx_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        # calculate fade n_time
        fade_ins = calc_fade(config.ctx_size)
        config.n_layer = len(fade_ins)
        # transformer blocks
        self.blocks = nn.Sequential(*[FadingBlock(config, fade_in) for fade_in in fade_ins])
        self.ln = LayerNorm(config.n_embd, config.bias)
        self.ff = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        # initialize weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, targets=None):
        b, t = x.size()
        device = x.device
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
        # get token and position embeddings
        tok_embd = self.tok_embd(x) # (B, T, C)
        pos_embd = self.pos_embd(pos) # (1, T, C)
        # add them up, apply dropout
        x = self.dropout(tok_embd + pos_embd) # (B, T, C)
        # apply transformer blocks then layer norm
        x = self.blocks(x) # (B, T, C)
        x = self.ln(x) # (B, T, C)

        if targets is not None:
            # if we are training
            logits = self.ff(x) # (B, T, V)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # if we are just doing inference
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # note: using list [-1] to preserve the time dimension
            logits = self.ff(x[:, [-1], :]) # (B, T, V)
            loss = None
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # crop or pad context if needed
            # if x.size(1) > self.config.ctx_size:
            #     x_in =  x[:, -self.config.ctx_size:]
            # else:
            #     x_in = F.pad(x, (self.config.ctx_size - x.size(1), 0), value=0)
            x_in =  x[:, -self.config.ctx_size:]
            # forward pass
            logits, _ = self.forward(x_in)
            # get logits of the last token and apply temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample token from probability distribution
            x_next = torch.multinomial(probs, num_samples=1)
            # append to sequence
            x = torch.cat((x, x_next), dim=1)
        return x
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
