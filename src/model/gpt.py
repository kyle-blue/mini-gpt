from typing import Optional
import torch

# Module with the building blocks for graphs (nns are actually DAGS)
from torch import nn
from torch.nn import functional as F
from model.attention_head import AttentionHead
from params import vocab_size, embedding_dim, block_size, device, combined_head_size


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding_table = nn.Embedding(block_size, embedding_dim)

        self.attention_head = AttentionHead(combined_head_size)
        # Final linear layer to convert outputs into logits
        self.lm_head = nn.Linear(combined_head_size, vocab_size)

    def forward(self, batch_x: torch.Tensor, batch_y: Optional[torch.Tensor]):
        B, T = batch_x.shape
        # logits are raw non-normalised predictions made by a model

        # batch_x is B, T (where B is batch and T is time / block_size)

        # B, T, C
        embeddings = self.embedding_table.forward(batch_x)

        # T, C
        # Have to use T directly here instead of block size just in case its not a full block
        # Apparentlys its better to actually pad this so block size is consistent. We would then have
        # to worry about optimisation of processing padded tokens, so lets leave that to another day
        pos_embeddings = self.positional_embedding_table.forward(
            torch.arange(T, device=device)
        )

        # B, T, C   +   T, C   = B, T, C  (torch does broadcasting)
        summed_embeddings = embeddings + pos_embeddings

        values = self.attention_head.forward(summed_embeddings)
        logits = self.lm_head.forward(values)

        if batch_y is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), batch_y.view(B * T))

        return logits, loss

    def generate(self, batch_x: torch.Tensor, max_tokens: int):
        for _ in range(max_tokens):
            logits, _ = self.forward(batch_x[:, -block_size:], None)
            last_logits = logits[:, -1, :]
            probability_dist = F.softmax(last_logits, dim=1)
            prediction = torch.multinomial(probability_dist, num_samples=1)
            batch_x = torch.cat((batch_x, prediction), dim=1)
        return batch_x
