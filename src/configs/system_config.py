from dataclasses import dataclass, field
from src.data.prompt_manager import PromptVersion

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

    prompt_version: PromptVersion = PromptVersion.V1

    # only inference
    save_dir: str = "" # if will be set by the function
    adapter_dir: str = "lora_adapter" # it will be set by the function
    test_result_dir: str = "test_results" # it will be set by the function

