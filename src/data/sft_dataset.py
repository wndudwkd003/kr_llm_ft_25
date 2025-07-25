import torch
from src.data.prompt_manager import PromptManager
from src.data.base_dataset import BaseDataset, make_chat

DEBUG = False

class SFTDataset(BaseDataset):
    def process_example(self, example):
        # 질문 길이 제한 적용
        question_text = example.get("input", {}).get("question", "")
        question_len = len(question_text.replace(" ", ""))  # 공백 제외

        # 데이터 질문 길이 제한
        self.data_question_length_limit = self.config_manager.system.data_question_length_limit

        if self.data_question_length_limit != -1 and question_len > self.data_question_length_limit:
            print(f"Skipping example due to question length: {question_len} > {self.data_question_length_limit}")
            return None

        # 시스템 프롬프트 가져오기
        system_prompt = PromptManager.get_system_prompt(self.config_manager.system.prompt_version)

        # 사용자 프롬프트 생성
        user_prompt = make_chat(
            example["input"],
            config_manager=self.config_manager,
        )

        # 메시지 구성
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if DEBUG: print(message)


        # 소스 토크나이즈
        source = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False
        )

        raw_output = example.get("output", "")
        if isinstance(raw_output, dict):
            target_text = raw_output.get("answer", "")
        else:
            target_text = raw_output


        if not target_text:
            return None

        # EOS 토큰 추가
        target_text = target_text + self.tokenizer.eos_token

        if DEBUG: print(target_text)  # 디버깅용 출력

        # 타겟 토크나이즈
        target = self.tokenizer(
            target_text,
            return_attention_mask=False,
            add_special_tokens=False,
            return_tensors="pt"
        )
        target["input_ids"] = target["input_ids"].type(torch.int64)

        input_ids = torch.concat((source[0], target["input_ids"][0]))
        labels = torch.concat((
            torch.LongTensor([self.IGNORE_INDEX] * source[0].shape[0]),
            target["input_ids"][0]
        ))

        return {
            "input_ids": input_ids,
            "labels": labels
        }


