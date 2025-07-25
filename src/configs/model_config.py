from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    model_id: str = ""
    dtype: str|torch.dtype = ""
    max_seq_length: int = 4096

    # only train
    early_stopping: int = 3
    early_stopping_threshold: float = 0.001

    # only inference
    max_new_tokens: int = 512
    do_sample: bool = False
