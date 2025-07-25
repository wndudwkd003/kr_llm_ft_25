import os
from typing import TypeVar
from datasets import Dataset as HFDataset          # ★ 추가
from src.configs.config_manager import ConfigManager
from src.data.dataset_factory import DatasetFactory
from src.data.dpo_dataset import DPODataset
from datasets import Dataset as HFDataset

T = TypeVar("T")

def _ensure_hf(ds):
    """torch Dataset → 🤗 Dataset 변환 (이미 HF면 그대로)"""
    if hasattr(ds, "map"):          # HFDataset은 map/shuffle 메서드가 있음
        return ds
    return HFDataset.from_list(list(ds))

def prepare_dataset(config_manager: ConfigManager, tokenizer, task_type: str = "sft"):
    prompt_version = config_manager.system.prompt_version
    print("Current prompt version:", prompt_version)

    data_root = config_manager.system.data_raw_dir

    if task_type.lower() == "dpo":
        # ────────────────────────────────────────────────
        # ① torch‑style DPODataset 로드
        # ────────────────────────────────────────────────
        train_ds_torch = DPODataset(
            fname=os.path.join(data_root, "train.json"),
            tokenizer=tokenizer,
            max_length=config_manager.model.max_seq_length,
        )
        eval_ds_torch  = DPODataset(
            fname=os.path.join(data_root, "dev.json"),
            tokenizer=tokenizer,
            max_length=config_manager.model.max_seq_length,
        )

        # ② 🤗 Dataset 으로 변환
        train_ds = HFDataset.from_list(list(train_ds_torch))
        eval_ds  = HFDataset.from_list(list(eval_ds_torch))

        return train_ds, eval_ds

    # 공통 인자 정리
    common_args = dict(
        dataset_type=task_type,
        tokenizer=tokenizer,
        config_manager=config_manager,
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

    # 🔑 DPO일 때는 HF Dataset으로 변환
    if task_type.lower() == "dpo":
        train_ds = _ensure_hf(train_ds)
        eval_ds  = _ensure_hf(eval_ds)

    return train_dataset, eval_dataset


def prepare_test_dataset(config_manager: ConfigManager, tokenizer, data_class: type[T]):
    test_dataset = data_class(
        fname=os.path.join(config_manager.system.data_raw_dir, "test.json"),
        tokenizer=tokenizer
    )

    return test_dataset
