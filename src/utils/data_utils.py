import os
from typing import TypeVar
from src.configs.config_manager import ConfigManager

T = TypeVar('T')

def prepare_dataset(config_manager: ConfigManager, tokenizer, data_class: type[T]):
    train_dataset = data_class(
        fname=os.path.join(config_manager.system.data_raw_dir, "train.json"),
        tokenizer=tokenizer
    )

    eval_dataset = data_class(
        fname=os.path.join(config_manager.system.data_raw_dir, "dev.json"),
        tokenizer=tokenizer
    )

    return train_dataset, eval_dataset


def prepare_test_dataset(config_manager: ConfigManager, tokenizer, data_class: type[T]):
    test_dataset = data_class(
        fname=os.path.join(config_manager.system.data_raw_dir, "test.json"),
        tokenizer=tokenizer
    )

    return test_dataset
