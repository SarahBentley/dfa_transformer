from dataclasses import dataclass

@dataclass 
class ModelConfig:
    vocab_size: int # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 40
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    pad_idx: int = 0
    MLP_width: int = None

@dataclass
class GTConfig(ModelConfig):
    block_size: int = 50

@dataclass
class WGTConfig(ModelConfig):
    window_size: int = 2

@dataclass
class RWGTConfig(WGTConfig):
    window_size: int = 2
    max_seq: int = 2048

@dataclass
class TrainConfig:
    dataloader: object
    num_epochs: int = 10
    lr: float = 0.001
    num_batches: int = 100
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    batch_size: float = 10

    
    
