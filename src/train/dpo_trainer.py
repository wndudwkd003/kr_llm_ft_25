#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unsloth + TRL 기반 DPO 학습 클래스 예시

사용법 개요
-----------
cm = ConfigManager(...)  # 기존과 동일한 설정 객체
trainer = UnslothDPOTrainer(cm)
trainer.setup_model()                # 모델/토크나이저/LoRA 세팅
train_ds, eval_ds = trainer.prepare_dataset()  # DPO용 데이터셋 로드
stats = trainer.train(train_ds, eval_ds)       # 학습 수행
trainer.save()                      # 어댑터/토크나이저 저장

필요 ConfigManager 키 예시
--------------------------
cm.model.model_id          : str  (HF 모델 ID)
cm.model.max_seq_length    : int
cm.model.dtype             : str  ("bfloat16" | "float16" | "float32" 등)

cm.lora.r                  : int
cm.lora.target_modules     : list[str]
cm.lora.lora_alpha         : int
cm.lora.lora_dropout       : float
cm.lora.bias               : str   ("none", "lora_only", ...)

cm.system.seed             : int
cm.system.data_raw_path    : str   (데이터 경로)
cm.system.output_dir       : str   (모델 저장 경로)

cm.dpo                     : dict  (transformers.TrainingArguments 에 들어갈 인자들)
  ├─ "output_dir": str
  ├─ "per_device_train_batch_size": int
  ├─ "per_device_eval_batch_size": int
  ├─ "learning_rate": float
  ├─ ...

cm.dpo_hparams             : dict (DPOTrainer 전용 하이퍼파라미터)
  ├─ "beta": float            # DPO 온도
  ├─ "loss_type": str        # "sigmoid" | "ipo" | "hinge" 등 (TRL>=0.9)
  ├─ "label_smoothing": float
  ├─ "reference_free": bool   # True 시 ref model 없이 학습
  └─ 기타 DPO 관련 인자

데이터 포맷 (train.json / dev.json)
-----------------------------------
각 라인은 아래 키를 포함한 JSON 객체:
{
  "prompt":   "질문 또는 시스템 프롬프트",
  "chosen":   "선호되는(정답) 응답",
  "rejected": "덜 선호되는(오답) 응답"
}

주요 의존 라이브러리
--------------------
unsloth>=0.6, transformers>=4.41, trl>=0.9
"""
import unsloth
from __future__ import annotations
import os
import json
from typing import Tuple, Optional, List, Dict

from unsloth import FastLanguageModel
from trl import DPOTrainer, DPODataCollatorWithPadding
from transformers import TrainingArguments, PreTrainedTokenizerBase
from torch.utils.data import Dataset

from src.data.dpo_dataset import DPODataset
from src.configs.config_manager import ConfigManager
from src.data.dpo_dataset import DataCollatorForDPODataset


class UnslothDPOTrainer:
    """Unsloth + TRL DPO 학습 래퍼 클래스."""

    def __init__(self, config_manager: ConfigManager):
        self.cm: ConfigManager = config_manager
        self.model = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.trainer: Optional[DPOTrainer] = None
        self.ref_model = None  # 필요 시 수동 설정 가능

    # ------------------------------------------------------------------
    # 1) 모델/토크나이저 준비 & LoRA 적용
    # ------------------------------------------------------------------
    def setup_model(self):
        """FastLanguageModel 로드 및 LoRA 설정."""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.cm.model.model_id,
            max_seq_length=self.cm.model.max_seq_length,
            dtype=self.cm.model.dtype,
            load_in_4bit=False,
            load_in_8bit=False,
            full_finetuning=False,
        )

        # pad_token 없으면 eos로 세팅
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # LoRA 적용
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.cm.lora.r,
            target_modules=self.cm.lora.target_modules,
            lora_alpha=self.cm.lora.lora_alpha,
            lora_dropout=self.cm.lora.lora_dropout,
            bias=self.cm.lora.bias,
            random_state=self.cm.system.seed,
        )

    # ------------------------------------------------------------------
    # 2) 데이터셋 준비
    # ------------------------------------------------------------------
    def prepare_dataset(self) -> Tuple[DPODataset, DPODataset]:
        """train/dev DPO 데이터셋 로드.

        Returns:
            (train_dataset, eval_dataset)
        """
        train_path = os.path.join(self.cm.system.data_raw_path, "train.json")
        dev_path = os.path.join(self.cm.system.data_raw_path, "dev.json")

        train_dataset = DPODataset(
            fname=train_path,
            tokenizer=self.tokenizer,
            max_length=self.cm.model.max_seq_length,
        )
        eval_dataset = DPODataset(
            fname=dev_path,
            tokenizer=self.tokenizer,
            max_length=self.cm.model.max_seq_length,
        )
        return train_dataset, eval_dataset

    # ------------------------------------------------------------------
    # 3) 학습 루틴
    # ------------------------------------------------------------------
    def train(self, train_dataset: DPODataset, eval_dataset: DPODataset):
        """DPOTrainer를 이용한 학습 수행."""
        training_args = TrainingArguments(**self.cm.dpo)

        # TRL에서 제공하는 전용 Collator (label 토큰 분리 등)
        data_collator = DataCollatorForDPODataset(
            tokenizer=self.tokenizer,
            max_length=self.cm.model.max_seq_length,
            pad_to_multiple_of=8,
        )

        # DPO 하이퍼파라미터 (beta, loss_type 등)
        dpo_kwargs = self.cm.dpo_hparams.copy() if hasattr(self.cm, "dpo_hparams") else {}

        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,  # None일 경우 내부에서 복제하여 reference model로 사용
            args=training_args,
            beta=dpo_kwargs.get("beta", 0.1),
            loss_type=dpo_kwargs.get("loss_type", "sigmoid"),
            label_smoothing=dpo_kwargs.get("label_smoothing", 0.0),
            reference_free=dpo_kwargs.get("reference_free", False),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        train_result = self.trainer.train()
        return train_result

    # ------------------------------------------------------------------
    # 4) 저장
    # ------------------------------------------------------------------
    def save(self, output_dir: Optional[str] = None, merge_adapter: bool = False):
        """모델/어댑터/토크나이저 저장.

        Args:
            output_dir (str): 저장 경로. None이면 cm.dpo["output_dir"] 사용
            merge_adapter (bool): True면 LoRA weight를 base model에 병합 후 저장
        """
        save_dir = output_dir or self.cm.dpo.get("output_dir", self.cm.system.output_dir)
        os.makedirs(save_dir, exist_ok=True)

        if merge_adapter:
            # Unsloth는 FastLanguageModel.save_pretrained 을 사용해 병합 가능
            FastLanguageModel.save_pretrained(
                self.model,
                save_dir,
                tokenizer=self.tokenizer,
                merge_lora=True,
            )
        else:
            # 어댑터만 저장
            self.trainer.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)

    # ------------------------------------------------------------------
    # 5) (선택) ref_model을 외부에서 제공하고 싶을 때
    # ------------------------------------------------------------------
    def load_reference_model(self, model_path: Optional[str] = None):
        """Reference model을 명시적으로 로드하고 싶을 때 사용.

        Args:
            model_path (str): HF 모델 ID 또는 로컬 경로. None이면 self.cm.model.model_id 재사용.
        """
        if model_path is None:
            model_path = self.cm.model.model_id

        ref_model, _ = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.cm.model.max_seq_length,
            dtype=self.cm.model.dtype,
            load_in_4bit=False,
            load_in_8bit=False,
            full_finetuning=False,
        )
        # LoRA 미적용 / Gradient 비활성화로 고정
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model = ref_model