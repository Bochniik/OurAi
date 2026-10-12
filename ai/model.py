import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(
            config.d_model,
            3 * config.d_model,
            bias=False
        )

        self.proj = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1, 2)

        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True
        )

        attention = attention.transpose(1, 2).contiguous()

        attention = attention.view(
            batch_size,
            seq_len,
            d_model
        )

        return self.proj(attention)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        hidden_size = 4 * config.d_model

        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)

        self.norm2 = nn.LayerNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))

        return x


class OurAI(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        self.position_embedding = nn.Embedding(
            config.context_length,
            config.d_model
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)

        self.output_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        # Token embedding a output používajú rovnaké váhy.
        self.output_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, input_ids, targets=None):
        batch_size, seq_len = input_ids.shape

        if seq_len > self.config.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"context length {self.config.context_length}"
            )

        positions = torch.arange(
            0,
            seq_len,
            device=input_ids.device
        )

        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)

        x = token_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.output_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss