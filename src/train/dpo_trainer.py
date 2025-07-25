import os
from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig
from transformers import TrainingArguments, EarlyStoppingCallback
from src.train.base_trainer import BaseTrainer
from dataclasses import asdict
from datasets import Dataset as HFDataset
from peft import PeftModel


class UnslothDPOTrainer(BaseTrainer):
    def setup_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.cm.model.model_id,
            max_seq_length=self.cm.model.max_seq_length,
            dtype=self.cm.model.dtype,
            load_in_4bit=False,
            load_in_8bit=False,
            full_finetuning=False,
        )

        self.tokenizer_setup()

        # DPO 전략: Pre-trained 모델에 SFT Adapter를 병합 -> DPO를 위해 새로운 Adapter를 병합된 모델에 추가
        sft_adapter_dir = os.path.join(
            self.cm.system.sft_model_for_dpo,
            self.cm.system.adapter_dir
        )

        if os.path.exists(sft_adapter_dir):
            print(f"Merging SFT adapter from {sft_adapter_dir}")
            # 1. SFT 어댑터 로드
            # self.model.load_adapter(sft_adapter_dir)

            peft_model = PeftModel.from_pretrained(
                self.model,
                sft_adapter_dir,
                is_trainable=False,
            )

            merged_model = peft_model.merge_and_unload()

            merged_dir = os.path.join(sft_adapter_dir, "merged")
            os.makedirs(merged_dir, exist_ok=True)

            merged_model.save_pretrained(merged_dir)
            self.tokenizer.save_pretrained(merged_dir)
            print(f"[DPO] SFT adapter merged & saved at: {merged_dir}")

            # 2) merge된 가중치를 다시 불러와 DPO용 fresh LoRA를 붙인다
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                merged_dir,
                max_seq_length=self.cm.model.max_seq_length,
                dtype=self.cm.model.dtype,
                load_in_4bit=False,
                load_in_8bit=False,
                full_finetuning=False,
            )

            self.tokenizer_setup()
            print("SFT adapter merged successfully")

            """

            todo: 여기 기존의 모델이 가중치 수가 변하는지 확인 해야 함. 그리고 모델 불러오는 중간에 안쓰는 용량 차지하는게 아닌지 확인 해야 함

            """

        # DPO를 위한 새로운 LoRA 어댑터 추가
        print("Adding new LoRA adapter for DPO training")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.cm.lora.r,
            target_modules=self.cm.lora.target_modules,
            lora_alpha=self.cm.lora.lora_alpha,
            lora_dropout=self.cm.lora.lora_dropout,
            bias=self.cm.lora.bias,
            random_state=self.cm.system.seed,
        )

        print(f"Model setup complete. Total parameters: {self.model.num_parameters():,}")
        print(f"Trainable parameters: {self.model.num_parameters(only_trainable=True):,}")

    def train(self, train_dataset: HFDataset, eval_dataset: HFDataset):
        dpo_dict = asdict(self.cm.dpo)
        training_args = DPOConfig(**dpo_dict)

        callbacks = [EarlyStoppingCallback(
            early_stopping_patience=self.cm.model.early_stopping,
            early_stopping_threshold=self.cm.model.early_stopping_threshold
        )]if self.cm.model.early_stopping else None


        self.trainer = DPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            args=training_args,
        )

        trainer_stats = self.trainer.train()
        return trainer_stats


    def save_adapter(self, save_path:str|None = None):
        """LoRA 어댑터 저장"""
        if save_path is None:
            save_path = os.path.join(self.cm.dpo.output_dir, self.cm.system.adapter_dir)

        self.model.save_pretrained(save_path)
        # self.tokenizer.save_pretrained(save_path)

        # 설정 파일도 함께 저장
        self.cm.update_config("system", {"hf_token": ""}) # remove token for security
        self.cm.save_all_configs(os.path.join(self.cm.dpo.output_dir, "configs"))

        print(f"Adapter saved to: {save_path}")
        return save_path
