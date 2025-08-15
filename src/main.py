from typing import List, Tuple, cast
from enum import Enum

import torch
from torch._prims_common import Tensor

from model.gpt import GPT
from params import (
    data_text,
    batch_size,
    chars,
    block_size,
    device,
    eval_interval,
    eval_iters,
    learning_rate,
    training_iters,
)


class Split(Enum):
    TRAINING = "training"
    VALIDATION = "validation"


def main():
    if device == "cuda":
        print("Cuda detected. Using GPU")
    else:
        print("Cuda not detected. Using CPU")

    # Extremely simple character level tokenizer
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}

    def encode(characters: str):
        return [stoi[c] for c in characters]

    def decode(integers: List[int]):
        return "".join([itos[i] for i in integers])

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
                _, loss = cast(Tuple[Tensor, Tensor], model.forward(xb, yb))
                losses[i] = loss

            out[split] = losses.mean()
        model.train()
        return out

    def get_batch(split: Split):
        data = train_data if split == Split.TRAINING else val_data
        ix = torch.randint(
            len(data) - block_size, (batch_size,)
        )  # tensor with block_size random numbers
        x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
        # +1 works since both randint and [:] are have exclusive top ranges, so it won't overflow
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
        return x, y

    model = GPT()
    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for i in range(training_iters):
        if i % eval_interval == 0:
            losses = estimate_loss()
            print(
                f"Training: {losses[Split.TRAINING]:.4f} --- Validation: {losses[Split.VALIDATION]:.4f}"
            )

        xb, yb = get_batch(Split.TRAINING)
        _, loss = cast(Tuple[Tensor, Tensor], model.forward(xb, yb))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

    start_x = torch.zeros((1, 1), dtype=torch.long, device=device)
    tokens = model.generate(start_x, 1000)
    print(decode(tokens.tolist()[0]))


if __name__ == "__main__":
    main()
