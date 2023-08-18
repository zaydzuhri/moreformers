"""
LLaMa Model definition, based on the galatolofederico/vanilla-llama implementation, but simpler.
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle

log_mode = 0 # 0: no log, 1: log attention matrix, 2: log block activations
if log_mode == 1:
    # create a file to log the attention matrix
    log_file = open('logs/attention_log.pkl', 'wb') # open the file in write binary mode

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs).float(), freqs.float())  # complex64
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    shape = [d if i == 1 or i == xq_.ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
    freqs_cis = freqs_cis.view(*shape).to(xq_.device)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out, xk_out

# Multi-head attention with KV cache
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ctx_size = config.ctx_size
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.out = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.register_buffer('cache_k', torch.zeros((config.batch_size, config.ctx_size, config.n_head, self.head_size)))
        self.register_buffer('cache_v', torch.zeros((config.batch_size, config.ctx_size, config.n_head, self.head_size)))
        self.register_buffer('tril', torch.tril(torch.ones(config.ctx_size, config.ctx_size)))

    def forward(self, x, freqs_cis, is_training):
        B, T, C = x.shape

        q = self.query(x) # (B, T, C)
        q = q.view(B, T, self.n_head, self.head_size) # (B, T, H, C/H)
        k = self.key(x)
        k = k.view(B, T, self.n_head, self.head_size)
        v = self.value(x)
        v = v.view(B, T, self.n_head, self.head_size)
        
        # apply rotary embeddings
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # compute attention "affinities", scale, mask, and softmax
        q = q.transpose(1, 2) # (B, H, T, C/H)
        k = k.transpose(1, 2) # (B, H, T, C/H)
        v = v.transpose(1, 2) # (B, H, T, C/H)
        att = q @ k.transpose(2, 3) # (B, H, T, C/H) @ (B, H, C/H, T) -> (B, H, T, T)
        att = att / math.sqrt(self.head_size) # scale
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # mask
        att = F.softmax(att, dim=-1) # softmax
        if log_mode == 1 and not is_training:
            # write the attention matrix to the log file using pickle
            pickle.dump({'batch_size': B, 'seq_length': T, 'head_size': C, 'att_matrix': att.cpu().numpy()}, log_file) # convert the tensor to numpy array and dump it as a dictionary

        # apply attention to value projection, concat heads, and project
        out = att @ v # (B, H, T, T) @ (B, H, T, C/H) -> (B, H, T, C/H)
        out = out.transpose(1, 2) # (B, T, H, C/H)
        out = out.flatten(2) # (B, T, C)
        out = self.out(out) # (B, T, C)
        return out

# Feedforward layer
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.n_embd * 4
        self.lin1 = nn.Linear(config.n_embd, hidden_size, bias=config.bias)
        self.lin2 = nn.Linear(hidden_size, config.n_embd, bias=config.bias)
        self.lin3 = nn.Linear(config.n_embd, hidden_size, bias=config.bias)
    
    def forward(self, x):
        return self.lin2(F.silu(self.lin1(x)) * self.lin3(x))

# Transformer block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.norm1 = RMSNorm(config.n_embd)
        self.ff = FeedForward(config)
        self.norm2 = RMSNorm(config.n_embd)
    
    def forward(self, x, freqs_cis, is_training):
        out = x + self.attn(self.norm1(x), freqs_cis, is_training)
        out = out + self.ff(self.norm2(out))
        return out

# LLaMA model
class LLaMA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.freqs_cis = precompute_freqs_cis(config.n_embd // config.n_head, config.ctx_size * 2)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.lin = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        # if log_mode == 2:
        #     # create a file to log activations
        #     self.log_file = open('logs/llama_log.pkl', 'wb') # open the file in write binary mode
    
    def forward(self, x, targets=None):
        is_training = targets is not None
        B, T = x.shape
        x = self.tok_emb(x) # (B, T, C)
        freqs_cis = self.freqs_cis[:T] # (T, T, 2)
        # i = 0
        for block in self.blocks:
            x = block(x, freqs_cis, is_training)
            # if log_mode == 2 and not is_training:
            #     # write the activations to the log file using pickle
            #     pickle.dump({'layer': i, 'block_activations': x.cpu().numpy()}, self.log_file)
            #     i += 1
        x = self.norm(x)

        if is_training:
            # if we are training
            logits = self.lin(x) # (B, T, V)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # if we are just doing inference
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # note: using list [-1] to preserve the time dimension
            logits = self.lin(x[:, [-1], :]) # (B, T, V)
            loss = None
        return logits, loss
    
    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # crop context if needed
            x_crop = x if x.size(1) <= self.config.ctx_size else x[:, -self.config.ctx_size:]
            # forward pass
            logits, _ = self.forward(x_crop)
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

