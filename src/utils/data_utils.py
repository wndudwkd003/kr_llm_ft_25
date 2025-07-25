import os
from typing import TypeVar
from src.configs.config_manager import ConfigManager
from src.data.dataset_factory import DatasetFactory

T = TypeVar('T')

def prepare_dataset(config_manager: ConfigManager, tokenizer, task_type: str = "sft"):
    prompt_version = config_manager.system.prompt_version
    print("Current prompt version:", prompt_version)

    # 공통 인자 정리
    common_args = dict(
        dataset_type=task_type,
        tokenizer=tokenizer,
        prompt_version=prompt_version,
        data_question_length_limit=config_manager.system.data_question_length_limit,
    )

    train_dataset = DatasetFactory.create_dataset(
        fname=os.path.join(config_manager.system.data_raw_dir, "train.json"),
        data_shuffle=config_manager.system.data_shuffle,
        **common_args
    )

    eval_dataset = DatasetFactory.create_dataset(
        fname=os.path.join(config_manager.system.data_raw_dir, "dev.json"),
        data_shuffle=False, # 평가 데이터는 셔플하지 않음
        **common_args
    )

    return train_dataset, eval_dataset


def prepare_test_dataset(config_manager: ConfigManager, tokenizer, data_class: type[T]):
    test_dataset = data_class(
        fname=os.path.join(config_manager.system.data_raw_dir, "test.json"),
        tokenizer=tokenizer
    )

    return test_dataset
