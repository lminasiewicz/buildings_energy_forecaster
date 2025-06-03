import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoder(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Based on the paper "Attention Is All You Need" p.6 (section 3.5 "Positional Encoding")
        # https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(1)
        
        # Register a part of the module state that is not to be trained
        self.register_buffer("positional_encoding", encoding)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.positional_encoding[x.size(0), :])
