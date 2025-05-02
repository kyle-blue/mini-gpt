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
batch_size = 32
block_size = 8
training_iters = 10000
eval_iters = 100
eval_interval = 500
learning_rate = 1e-3  # Apparently attention blocks can't handle larger learning rates. TODO: Maybe look into why?
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embed = 32  # Number of embedding dimensions
vocab_size = len(chars)
num_heads = 4  # Head size of each head will be n_embed // num_heads
