from torch.nn.modules import LayerNorm
from params import num_heads, n_embed
import torch
from torch import nn

from model.attention_head import MultiHeadAttention
from model.feed_forward import FeedForward


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        head_size = n_embed // num_heads
        self.mha = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward()
        self.ln1 = LayerNorm(n_embed)
        self.ln2 = LayerNorm(n_embed)
        # -> Takes in embeddings
        # Feeds into MultipleHeads
        # Makes inferences from the outputs (FeedForward)
        #

    def forward(self, batch_x: torch.Tensor):
        # Residual connections to simply make each layer add useful information upon inputs, rather than creating new info
        # Also helps with vanishing gradient problem, maintains a steady gradient throughout the network TODO: Look into backprop to fully understand this
        batch_x += self.mha.forward(self.ln1(batch_x))
        batch_x += self.ffwd.forward(self.ln2(batch_x))
        return batch_x
