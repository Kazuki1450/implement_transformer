import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k

    def forward(
        self,
        q: torch.Tensor[float],
        k: torch.Tensor[float],
        v: torch.Tensor[float],
        mask: torch.Tensor[bool],
    ) -> torch.Tensor:

        attention_weight: torch.Tensor = torch.matmul(
            q, torch.transpose(k, 1, 2)
        ) / np.sqrt(self.d_k)
        if mask is not None:
            attention_weight = attention_weight.data.masked_fill_(
                mask, -torch.finfo(torch.float).max
            )
        attention_weight = F.softmax(attention_weight, dim=2)
        return torch.matmul(attention_weight, v)
