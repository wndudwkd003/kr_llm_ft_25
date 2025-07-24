from dataclasses import dataclass, field

@dataclass
class SystemConfig:
    is_train: bool = True
    additional_info: str = "" # it will be concatenated to other configs
    seed: int = 42
    hf_token: str = "" # it will be set by the function
    use_lora: bool = True
    use_qlora: bool = False
    data_raw_dir: str = "data/raw"
    data_rag_dir: str = "data/rag"

    # only inference
    save_dir: str = "" # if will be set by the function
    adapter_dir: str = "lora_adapter" # it will be set by the function
    test_result_dir: str = "test_results" # it will be set by the function

