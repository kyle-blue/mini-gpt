import torch
from torch import nn
from params import n_embed, block_size


class AttentionHead(torch.nn.Module):
    tril: torch.Tensor

    def __init__(self, head_size: int):
        super().__init__()
        self.key_layer = nn.Linear(n_embed, head_size)
        self.query_layer = nn.Linear(n_embed, head_size)
        self.value_layer = nn.Linear(n_embed, head_size)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, batch_x: torch.Tensor):
        _, T, _ = batch_x.shape  # B, T, C
        keys = self.key_layer.forward(batch_x)  # B, T, C (but c is head_size
        queries = self.query_layer.forward(batch_x)
        values = self.value_layer.forward(batch_x)

        # B, T, C   @   B, C, T  =   B, T, T
        wei = (queries @ keys.transpose(-2, -1)) / values.shape[-1] ** 0.5
        wei = torch.softmax(
            wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")), dim=-1
        )

        out = wei @ values

        return out


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [AttentionHead(head_size) for _ in range(num_heads)]
        )
        # I believe this is actually mainly a layer to convert / project between dimensions when head_size * num_heads is not equal to n_embed
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, batch_x: torch.Tensor):
        output = torch.cat([head.forward(batch_x) for head in self.heads], dim=-1)
        output = self.proj.forward(output)
        return output
