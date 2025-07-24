import os, datetime
from src.configs.config_manager import ConfigManager

def get_output_dir(cm: ConfigManager, train_type: str = "dpo"):
    base_path = cm.sft.output_dir
    converted_model_id = slash_remove(cm.model.model_id)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"{current_time}_{converted_model_id}_r_{cm.lora.r}_ra_{cm.lora.lora_alpha}_rd_{cm.lora.lora_dropout}_{train_type}"
    return os.path.join(base_path, save_dir)

def slash_remove(path: str):
    return path.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
