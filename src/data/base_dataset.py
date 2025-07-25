import json, torch, random
from typing import Any
from abc import ABC, abstractmethod
from src.data.prompt_manager import PromptVersion, PromptManager

class BaseDataset(ABC):
    def __init__(
        self,
        fname: str,
        tokenizer,
        prompt_version: PromptVersion = PromptVersion.V0,
        data_question_length_limit: int = 512,
        data_shuffle: bool = False
    ):
        self.fname = fname
        self.tokenizer = tokenizer
        self.prompt_version = prompt_version
        self.IGNORE_INDEX = -100
        self.data_question_length_limit = data_question_length_limit
        self.data_shuffle = data_shuffle

        # 데이터 저장용
        self.inp = []
        self.label = []

        # 데이터 로드 및 처리
        self._load_and_process_data()

    def _load_and_process_data(self):
        """데이터 로드 및 처리 파이프라인"""
        with open(self.fname, "r") as f:
            data = json.load(f)

            if self.data_shuffle:
                print(f"Shuffling data... {len(data)} examples")
                random.shuffle(data)
                print("Data shuffled.")

        for example in data:
            processed_example = self.process_example(example)
            if processed_example:  # None이 아닌 경우만 추가
                self.inp.append(processed_example["input_ids"])
                self.label.append(processed_example["labels"])

    @abstractmethod
    def process_example(
        self,
        example: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """
        각 데이터 타입별로 구현해야 하는 예제 처리 메서드

        Args:
            example: 원본 데이터 예제

        Returns:
            {"input_ids": tensor, "labels": tensor} 또는 None (스킵할 경우)
        """
        pass

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inp[idx],
            "labels": self.label[idx],
        }

def make_chat(inp, prompt_version: PromptVersion):
    """입력 데이터를 채팅 형식으로 변환하는 함수 (버전별 프롬프트 적용)"""

    # 버전에 맞는 instruction 가져오기
    instruction = PromptManager.get_instruction_for_type(prompt_version, inp.get('question_type', ''))

    # 기타 정보 생성 (question과 question_type 제외)
    other_info = {k: v for k, v in inp.items() if k not in ['question', 'question_type']}

    # 기타 정보가 있는 경우에만 추가
    chat_parts = [instruction]
    if other_info:
        info_list = ["[기타 정보]"]
        for key, value in other_info.items():
            if value is not None and value != "":
                info_list.append(f" {key}: {value}")
        chat_parts.append(" ".join(info_list))

    # 질문 추가
    chat_parts.append(f"[질문] {inp['question']}")

    # 최종 프롬프트 생성
    chat = " ".join(chat_parts)

    if False: print(chat)  # 디버깅용 출력

    return chat
