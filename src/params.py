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
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_dim = 32
vocab_size = len(chars)
