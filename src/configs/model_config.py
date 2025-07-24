from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    model_id: str = ""
    dtype: str|torch.dtype = ""
    use_flash_attention: bool = True
    max_seq_length: int = 4096

    # only train
    early_stopping: int = 3

    # only inference
    max_new_tokens: int = 512
    do_sample: bool = False
