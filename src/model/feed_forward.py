import torch
from torch import nn
from params import feed_forward_scale_up, n_embed, dropout_factor


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * feed_forward_scale_up),
            nn.ReLU(),
            nn.Linear(n_embed * feed_forward_scale_up, n_embed),
            nn.Dropout(dropout_factor),
        )

    def forward(self, batch_x: torch.Tensor):
        out = self.net(batch_x)
        return out
