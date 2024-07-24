from dataclasses import dataclass
import torch.nn as nn
import torch
from ..DFA.DFA import DFA

@dataclass
class TrainConfig:
    dataloader: object
    num_epochs: int = 10
    lr: float = 0.2
    seq_len: int = 10
    l1_penalty: float = 0.01