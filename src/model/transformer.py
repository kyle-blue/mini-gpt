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
        if head_size * num_heads != n_embed:
            raise Exception(
                f"num_heads ({num_heads}) is not a factor of n_embed ({n_embed}) which means there will be a shape mismatch when splitting features into multiple attention heads in multi-headed attention blocks:"
            )
        self.mha = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward()
        # Layer norm has gamma and beta learnable params per feature
        # Layer norm over batch norm as sequence lengths vary
        self.ln1 = LayerNorm(n_embed)
        self.ln2 = LayerNorm(n_embed)
        # -> Takes in embeddings
        # Feeds into MultipleHeads
        # Makes inferences from the outputs (FeedForward)
        #

    def forward(self, batch_x: torch.Tensor):
        # Residual connections to simply make each layer add useful information upon inputs, rather than creating new info
        # Also helps with vanishing gradient problem, maintains a steady gradient throughout the network TODO: Look into backprop to fully understand this
        norm1 = self.ln1.forward(batch_x)
        # Cannot use += since this would be an in-place operation.
        # This means that the original value of batch_x wouldn't otherwise be remembered in the graph
        batch_x = batch_x + self.mha.forward(norm1)
        norm2 = self.ln2.forward(batch_x)
        batch_x = batch_x + self.ffwd.forward(norm2)
        return batch_x
