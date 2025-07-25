import os
from typing import TypeVar
from src.configs.config_manager import ConfigManager
from src.data.dataset_factory import DatasetFactory

T = TypeVar('T')

def prepare_dataset(config_manager: ConfigManager, tokenizer, task_type: str = "sft"):
    prompt_version = config_manager.system.prompt_version
    print("Current prompt version:", prompt_version)

    train_dataset = DatasetFactory.create_dataset(
        dataset_type=task_type,
        fname=os.path.join(config_manager.system.data_raw_dir, "train.json"),
        tokenizer=tokenizer,
        prompt_version=prompt_version,
        data_question_length_limit=config_manager.system.data_question_length_limit
    )

    eval_dataset = DatasetFactory.create_dataset(
        dataset_type=task_type,
        fname=os.path.join(config_manager.system.data_raw_dir, "dev.json"),
        tokenizer=tokenizer,
        prompt_version=prompt_version,
        data_question_length_limit=config_manager.system.data_question_length_limit
    )

    return train_dataset, eval_dataset


def prepare_test_dataset(config_manager: ConfigManager, tokenizer, data_class: type[T]):
    test_dataset = data_class(
        fname=os.path.join(config_manager.system.data_raw_dir, "test.json"),
        tokenizer=tokenizer
    )

    return test_dataset
