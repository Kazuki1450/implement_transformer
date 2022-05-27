from typing import Optional

import torch
import torch.nn as nn

from .embedding import Embedding
from .feed_forward import FFN
from .multihead import MultiHeadAttention
from .positional_encoding import AddPositionalEncoding


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        heads_num: int,
        dropout_rate: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()

        self.multi_head_sa = MultiHeadAttention(d_model, heads_num)
        # アーキテクチャのfigureにはdropout層はないが、5.4章に利用したと記載されている
        self.dropout_sa = nn.Dropout(dropout_rate)
        self.layer_norm_sa = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.ffn = FFN(d_model, d_ff)
        self.dropout_ffn = nn.Dropout(dropout_rate)
        self.layer_norm_ffn = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.multi_head_sa(x, x, x, mask)
        out = self.dropout_sa(out)
        out = self.layer_norm_sa(out + x)

        res = out
        out = self.ffn(out)
        out = self.dropout_ffn(out)
        out = self.layer_norm_ffn(out + res)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        pad_idx: int,
        d_model: int,
        N: int,
        d_ff: int,
        heads_num: int,
        dropout_rate: float,
        layer_norm_eps: float,
        device: Optional[torch.device] = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, pad_idx)
        self.positional_encoding = AddPositionalEncoding(d_model, max_len, device)
        self.encoder_blocks = [
            TransformerEncoderBlock(
                d_model, d_ff, heads_num, dropout_rate, layer_norm_eps
            )
            for _ in range(N)
        ]

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.embedding(x)
        out = self.positional_encoding(out)
        for block in self.encoder_blocks:
            out = block(out, mask)
        return out
