from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    context_length: int = 512

    n_layers: int = 6
    n_heads: int = 5
    d_model: int = 320

    dropout: float = 0.1

    @property
    def head_dim(self):
        return self.d_model // self.n_heads