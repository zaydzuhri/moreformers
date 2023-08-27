# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import pickle


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    vocab_size: int = -1  # defined later by tokenizer
    ffn_dim_multiplier: int = 4
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f'{freqs_cis.shape} != {(x.shape[1], x.shape[-1])}'
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq)

    
class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        out_dim: int,
    ):
        super().__init__()

        self.w1 = torch.nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = torch.nn.Linear(
            hidden_dim, out_dim, bias=False
        )
        self.w3 = torch.nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Mix(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.max_seq_len = args.max_seq_len

        self.ff = FeedForward(
            dim=self.max_seq_len,
            hidden_dim=self.max_seq_len,
            out_dim=self.max_seq_len
        )

        self.wm = torch.nn.Linear(
            self.max_seq_len,
            1,
            bias=False
        )

        self.wo = torch.nn.Linear(
            args.dim, args.dim, bias=False
        )
        
        self.cache = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                args.dim
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        is_training: bool,
    ):
        bsz, seqlen, _ = x.shape # (B, T, C)
        # apply rotary embedding
        x = apply_rotary_emb(x, freqs_cis) # (B, T, C)
        xt = x.transpose(1, 2) # (B, C, T)
        # duplicate to (B, C, T, T)
        xt = xt.unsqueeze(-1).expand(-1, -1, -1, seqlen) # (B, C, T, T)
        # apply mask
        xt = xt * mask # (B, C, T, T)
        # apply feedforward
        xt = self.ff(xt) # (B, C, T, T)
        # apply wm
        out = self.wm(xt).squeeze(-1) # (B, C, T)
        out = out.transpose(1, 2) # (B, T, C)
        return self.wo(out)

class MixerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.mix = Mix(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.dim * args.ffn_dim_multiplier,
            out_dim=args.dim
        )
        self.layer_id = layer_id
        self.mix_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        is_training: bool,
    ):
        h = x + self.mix.forward(
            self.mix_norm(x), start_pos, freqs_cis, mask, is_training
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Mixer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = torch.nn.Embedding(
            params.vocab_size, params.dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(MixerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = torch.nn.Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim, self.params.max_seq_len * 2
        )

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None, start_pos: int = 0):
        is_training = targets is not None
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freq_start = min(start_pos, (self.params.max_seq_len*2) - 1)
        freqs_cis = self.freqs_cis[freq_start : freq_start + seqlen]

        mask = torch.full(
            (1, 1, self.params.max_seq_len, self.params.max_seq_len), float(0), device=tokens.device
        )
        mask = torch.triu(mask).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask, is_training)
        h = self.norm(h)
        output = self.output(h)
        
        if is_training:
            loss = F.cross_entropy(output.view(-1, output.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None

        return output, loss
    
    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0, top_k=None):
        bsz, seqlen = x.shape
        # crop x if longer than max_seq_len
        if seqlen > self.params.max_seq_len:
            x = x[:, -self.params.max_seq_len:]
            seqlen = self.params.max_seq_len
        total_len = seqlen + max_new_tokens
        prev_pos = 0
        for cur_pos in range(seqlen, total_len):
            logits, _ = self.forward(x[:, prev_pos:cur_pos], start_pos=prev_pos)
            logits = logits[:, -1]
            logits /= temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=1)
            prev_pos = cur_pos
        
        return x
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)