import os
import torch

# DATA LOADING
data_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../data/tiny-shakespeare.txt")
)
with open(data_path, "r") as file:
    data_text = file.read()

chars = sorted(list(set(data_text)))


# MANUAL SEED
torch.manual_seed(1234)

# HYPERPARAMETERS
batch_size = 128
block_size = 256
training_iters = 3000
eval_iters = 100
eval_interval = 500
learning_rate = 1e-3  # Apparently attention blocks can't handle larger learning rates. TODO: Maybe look into why?
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embed = 240  # Number of embedding dimensions
vocab_size = len(chars)
num_heads = 6  # Head size of each head will be n_embed // num_heads
feed_forward_scale_up = 4  # The multiplier by which we scale up the dimensions in the feed forward layer after each attention block to increase feature separation / extraction
num_layers = 6
dropout_factor = 0.2
