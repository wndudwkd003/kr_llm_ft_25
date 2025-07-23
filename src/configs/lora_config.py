from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.2
    target_modules: str|list = "all-linear"
    bias: str = "none"
    init_lora_weights: str = "xavier_uniform"
    use_dora: bool = False
    use_rslora: bool = False

