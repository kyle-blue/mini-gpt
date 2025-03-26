from enum import Enum
import os
import torch
from model import BigramModel


# HYPERPARAMETERS

torch.manual_seed(1234)
batch_size = 4
block_size = 8

class Split(Enum):
    TRAINING = "training"
    VALIDATION = "validation"


def main():
    data_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "../data/tiny-shakespeare.txt"))
    data_text = open(data_path, 'r').read()

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


    def get_batch(split: Split):
        data = train_data if split == Split.TRAINING else val_data
        ix = torch.randint(0, len(data - block_size), (batch_size,)) # tensor with block_size random numbers
        x = torch.stack([data[i:i+block_size] for i in ix])
        # +1 works since both randint and [:] are have exclusive top ranges, so it won't overflow
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y


    # xb, yb = get_batch(Split.TRAINING)
    
    # for i_seq in range(len(xb)):
    #     for i_token in range(len(xb[i_seq])):
    #         print(f"When input is {xb[i_seq][:i_token + 1]} target is {yb[i_seq][i_token]}")

    model = BigramModel(vocab_size)
    start_x = torch.tensor([[0]], dtype=torch.long)
    tokens = model.generate(start_x, 100)
    print(decode(tokens.tolist()[0]))

    

if __name__ == "__main__":
    main()