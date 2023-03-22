import torch
import math


def position_encoding_1d(d_model: int, length: int, base: float = 10000):
    assert d_model % 2 == 0, f"Cannot use sin/cos positional encoding with odd dim (got dim={d_model})"

    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(base) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe
