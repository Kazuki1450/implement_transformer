from typing import Optional

import torch
import torch.nn as nn

from layers.scaled_dot_prod import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.W_k = nn.parameter.Parameter(torch.tensor(h, d_model, self.d_k))
        self.W_q = nn.parameter.Parameter(torch.tensor(h, d_model, self.d_k))
        self.W_v = nn.parameter.Parameter(torch.tensor(h, d_model, self.d_v))
        self.scaled_dot_prod = ScaledDotProductAttention(self.d_k)
        self.linear = nn.Linear(h * self.d_v, d_model)

    def forward(
        self,
        q: torch.Tensor[float],
        k: torch.Tensor[float],
        v: torch.Tensor[float],
        mask_3d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        batch_size, seq_len = q.size(0), q.size(1)

        # 各headに渡すためにrepeatして、(head, batch, seq_len, d_model) にする
        q = q.repeat(self.h, 1, 1, 1)
        k = k.repeat(self.h, 1, 1, 1)
        v = v.repeat(self.h, 1, 1, 1)

        # (head, batch, seq_len, d_model) * (head, d_model, d_k) -> (head, batch, seq_len, d_k)
        q = torch.einsum("hijk,hkl->hijk", (q, self.W_q))
        # (head, batch, seq_len, d_model) * (head, d_model, d_k) -> (head, batch, seq_len, d_k)
        k = torch.einsum("hijk,hkl->hijl", (k, self.W_k))
        # (head, batch, seq_len, d_model) * (head, d_model, d_v) -> (head, batch, seq_len, d_v)
        v = torch.einsum("hijk,hkl->hijl", (v, self.W_v))

        # scaled_dot_prodへの入力として次元を揃えるためにhead*batchでbatchとみなしている。
        q = q.view(self.h * batch_size, seq_len, self.d_k)
        k = k.view(self.h * batch_size, seq_len, self.d_k)
        v = v.view(self.h * batch_size, seq_len, self.d_v)

        if mask_3d is not None:
            mask_3d = mask_3d.repeat(self.h, 1, 1)

        # 出力は (head*batch, seq_len, d_k)
        attention_output = self.scaled_dot_prod(q, k, v, mask_3d)

        # dim=0に沿ってattention_ouputをheadの個数に分割する
        attention_output = torch.chunk(attention_output, self.h, dim=0)

        # 分割されたh個のtensorをdim=2でcatして(batch, seq_len, d_model)に戻す。
        attention_output = torch.cat(attention_output, dim=2)

        # d_model%h==0でない場合はここでサイズが揃う意義もある。
        output = self.linear(attention_output)
        return output
