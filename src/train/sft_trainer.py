import os
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from src.configs.config_manager import ConfigManager
from src.data.sft_dataset import SFTDataset
from src.data.dataset import DataCollatorForSupervisedDataset


class UnslothSFTTrainer:
    def __init__(self, config_manager: ConfigManager):
        self.cm = config_manager
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.cm.model.model_id,
            max_seq_length=self.cm.model.max_seq_length,
            dtype=self.cm.model.dtype,
            load_in_4bit=False,
            load_in_8bit=False,
            full_finetuning=False,
        )

        # if tokenizer pad token is None, set it to eos token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.cm.lora.r,
            target_modules=self.cm.lora.target_modules,
            lora_alpha=self.cm.lora.lora_alpha,
            lora_dropout=self.cm.lora.lora_dropout,
            bias=self.cm.lora.bias,
            random_state=self.cm.system.seed,
        )

    def prepare_dataset(self):
        train_dataset = SFTDataset(
            fname=os.path.join(self.cm.system.data_raw_path, "train.json"),
            tokenizer=self.tokenizer
        )

        eval_dataset = SFTDataset(
            fname=os.path.join(self.cm.system.data_raw_path, "dev.json"),
            tokenizer=self.tokenizer
        )

        return train_dataset, eval_dataset


    def train(self, train_dataset: SFTDataset, eval_dataset: SFTDataset):
        training_args = TrainingArguments(**self.cm.sft)

        data_collator = DataCollatorForSupervisedDataset(
            tokenizer=self.tokenizer,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            args=training_args,
        )

        trainer_stats = self.trainer.train()
        return trainer_stats
