from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_id: str = ""
    dtype: str = ""
    use_flash_attention: bool = True

    # only train
    early_stopping_count: int = 3

    # only inference
    max_new_tokens: int = 512
    do_sample: bool = False
