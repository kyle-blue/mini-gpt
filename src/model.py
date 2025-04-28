from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F


class BigramModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        embedding_dim = vocab_size
        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, batch_x: torch.Tensor, batch_y: Optional[torch.Tensor]):
        # batch_x is B, T (where B is batch and T is time / block_size)
        logits = self.embedding_table.forward(
            batch_x
        )  # This will be B, T, C where C is embeddings

        if batch_y is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), batch_y.view(B * T))

        return logits, loss

    def generate(self, batch_x: torch.Tensor, max_tokens: int):
        for _ in range(max_tokens):
            logits, _ = self.forward(batch_x, None)
            last_logits = logits[:, -1, :]
            probability_dist = F.softmax(last_logits, dim=1)
            prediction = torch.multinomial(probability_dist, num_samples=1)
            batch_x = torch.cat((batch_x, prediction), dim=1)
        return batch_x
