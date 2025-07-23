from dataclasses import dataclass, field

@dataclass
class SystemConfig:
    is_train: bool = True
    additional_info: str = "" # it will be concatenated to other configs
    seed: int = 42
    devices: str = "0"
    hf_token: str = "" # it will be set by the function
    use_lora: bool = True
    use_qlora: bool = False
    data_raw_path: str = "/workspace/kli_llm/data/raw"
    data_rag_path: str = "/workspace/kli_llm/data/rag"

    # only inference
    loaded_model_path: str = "" # if current task is inference, must be set to the path of model to load

