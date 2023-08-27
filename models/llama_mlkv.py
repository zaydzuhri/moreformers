# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import pickle

log = False
if log:
    log_file = open('attention_log.pkl', 'wb')

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    n_kv_layers: int = 32
    max_batch_size: int = 32
    max_seq_len: int = 2048
    max_cache_batch_size: int = 256

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
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Project(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
    ):
        super().__init__()
        hidden_dim = dim * 4

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

class Attention(nn.Module):
    def __init__(self, has_kv: bool, args: ModelArgs):
        super().__init__()
        self.has_kv = has_kv
        self.max_seq_len = args.max_seq_len
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = torch.nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )

        if has_kv:
            self.wk = Project(
                args.dim,
                self.n_kv_heads * self.head_dim
            )
            self.wv = Project(
                args.dim,
                self.n_kv_heads * self.head_dim
            )

        self.wo = torch.nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        keys: Optional[torch.Tensor],
        values: Optional[torch.Tensor],
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        is_training: bool,
    ):
        bsz, seqlen, _ = x.shape
        xq = self.wq(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        
        if self.has_kv:
            xk, xv = self.wk(x), self.wv(x)
            xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

            if is_training:
                keys = xk
                values = xv
            else:
                if start_pos < self.max_seq_len:
                    keys[:bsz, start_pos : start_pos + seqlen] = xk
                    values[:bsz, start_pos : start_pos + seqlen] = xv
                else:
                    keys = torch.roll(keys, shifts=-seqlen, dims=1)
                    values = torch.roll(values, shifts=-seqlen, dims=1)
                    keys[:bsz, -seqlen:] = xk
                    values[:bsz, -seqlen:] = xv
            
        o_keys = keys
        o_values = values

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        if log:
            pickle.dump({'start_pos': start_pos, 'att_matrix': scores}, log_file)

        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output), o_keys, o_values


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = torch.nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = torch.nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = torch.nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, has_kv: bool, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(has_kv, args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        keys: Optional[torch.Tensor],
        values: Optional[torch.Tensor],
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        is_training: bool,
    ):
        h, keys, values = self.attention.forward(
            self.attention_norm(x), keys, values, start_pos, freqs_cis, mask, is_training
        )
        h = x + h
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out, keys, values


class LLaMAMLKV(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.n_kv_heads = params.n_heads if params.n_kv_heads is None else params.n_kv_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.head_dim = params.dim // params.n_heads

        self.tok_embeddings = torch.nn.Embedding(
            params.vocab_size, params.dim
        )

        # list layer indices that will host a kv head. first is always 0, rest is spread evenly until there are total n_kv_layers
        # example: n_layers=32, n_kv_layers=4 -> kv_layers=[0, 8, 16, 24]
        self.kv_layers = [0] + [
            int((i + 1) * (params.n_layers / params.n_kv_layers))
            for i in range(params.n_kv_layers - 1)
        ]

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, (layer_id in self.kv_layers), params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = torch.nn.Linear(
            params.dim, params.vocab_size, bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        self.cache_k = torch.zeros(
            (
                params.n_kv_layers,
                params.max_cache_batch_size,
                params.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                params.n_kv_layers,
                params.max_cache_batch_size,
                params.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None, start_pos: int = 0):
        is_training = targets is not None
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freq_start = min(start_pos, (self.params.max_seq_len*2) - 1)
        freqs_cis = self.freqs_cis[freq_start : freq_start + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
            )
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        self.cache_k = self.cache_k.to(h.device)
        self.cache_v = self.cache_v.to(h.device)
        keys = None
        values = None
        for layer in self.layers:
            i = layer.layer_id
            kv_i = max([ix for ix, l in enumerate(self.kv_layers) if l <= i])
            if not is_training:
                keys = self.cache_k[kv_i, :_bsz, : start_pos + seqlen]
                values = self.cache_v[kv_i, :_bsz, : start_pos + seqlen]

            h, keys, values = layer(h, keys, values, start_pos, freqs_cis, mask, is_training)

            if not is_training and i in self.kv_layers:
                self.cache_k[kv_i, :_bsz, : start_pos + seqlen] = keys
                self.cache_v[kv_i, :_bsz, : start_pos + seqlen] = values

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