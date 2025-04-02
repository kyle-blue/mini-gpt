from enum import Enum
import os
import torch
from model import BigramModel

torch.manual_seed(1234)

# HYPERPARAMETERS
batch_size = 32
block_size = 8
training_iters = 10000
eval_iters = 100
eval_interval = 500
embedding_size = 32
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Split(Enum):
    TRAINING = "training"
    VALIDATION = "validation"


def main():
    data_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../data/tiny-shakespeare.txt"))
    with open(data_path, 'r') as file:
        data_text = file.read()

    chars = sorted(list(set(data_text)))
    vocab_size = len(chars)

    # Extremely simple character level tokenizer
    stoi = { c:i for i,c in enumerate(chars) }
    itos = { i:c for i,c in enumerate(chars) }

    encode = lambda characters: [stoi[c] for c in characters]
    decode = lambda integers: ''.join([itos[i] for i in integers])

    # int64 datatype
    data = torch.tensor(encode(data_text), dtype=torch.long)

    train_split = int(0.9 * len(data))
    train_data = data[:train_split]
    val_data = data[train_split:]


    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for _, split in enumerate(Split):
            losses = torch.zeros(eval_iters)
            for i in range(eval_iters):
                xb, yb = get_batch(split)
                _, loss = model.forward(xb, yb)
                losses[i] = loss
            out[split] = losses.mean()
        model.train()
        return out

    def get_batch(split: Split):
        data = train_data if split == Split.TRAINING else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,)) # tensor with block_size random numbers
        x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
        # +1 works since both randint and [:] are have exclusive top ranges, so it won't overflow
        y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
        return x, y

    model = BigramModel(vocab_size)
    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for i in range(training_iters):
        if i % eval_iters == 0:
            losses = estimate_loss()
            print(f"Training: {losses[Split.TRAINING]:.4f} --- Validation: {losses[Split.VALIDATION]:.4f}")
        
        xb, yb = get_batch(Split.TRAINING)
        logits, loss = model.forward(xb, yb)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()       

    start_x = torch.tensor([[0]], dtype=torch.long, device=device)
    tokens = model.generate(start_x, 100)
    print(decode(tokens.tolist()[0]))

    

if __name__ == "__main__":
    main()