import torch
from torch import nn
from params import feed_forward_scale_up, n_embed


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * feed_forward_scale_up),
            nn.ReLU(),
            nn.Linear(n_embed * feed_forward_scale_up, n_embed),
        )

    def forward(self, batch_x: torch.Tensor):
        return self.net(batch_x)

