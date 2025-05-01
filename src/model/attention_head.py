import torch
from torch import nn
from params import embedding_dim, block_size


class AttentionHead(torch.nn.Module):
    def __init__(self, head_size: int):
        self.key_layer = nn.Linear(embedding_dim, head_size)
        self.query_layer = nn.Linear(embedding_dim, head_size)
        self.value_layer = nn.Linear(embedding_dim, head_size)
        self.tril = self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, batch_x: torch.Tensor):
        B, T, C = batch_x.shape
        keys = self.key_layer.forward(batch_x)  # B, T, C (but c is head_size
        queries = self.query_layer.forward(batch_x)
        values = self.value_layer.forward(batch_x)

        # B, T, C @ B, C, T  = B, T, T
        wei = (queries @ keys.transpose(-2, -1)) / values.shape[-1] ** 0.5
        wei = torch.softmax(wei.masked_fill(self.trill == 0, float("-inf")), dim=-1)

        out = wei @ values

        return out
