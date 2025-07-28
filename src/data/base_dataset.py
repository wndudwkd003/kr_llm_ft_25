import json, torch, random
from typing import Any
from abc import ABC, abstractmethod
from src.configs.config_manager import ConfigManager
from src.data.prompt_manager import PromptVersion, PromptManager

DEBUG = False  # 디버깅 모드 설정

OTHER_INFO_MAP = {
    "category": "카테고리",
    "topic_keyword": "키워드",
    "domain": "도메인",
    "question_type": "질문유형",
}

class BaseDataset(ABC):
    def __init__(
        self,
        fname: str,
        tokenizer,
        config_manager: ConfigManager,
        data_shuffle=False,
        task_type: str = "sft",  # ← task type 입력받기
    ):
        self.fname = fname
        self.tokenizer = tokenizer
        self.config_manager = config_manager
        self.IGNORE_INDEX = -100
        self.data_shuffle = data_shuffle
        self.task_type = task_type

        self.samples = []  # input_ids/labels 또는 prompt/chosen/rejected 통합 저장
        self._load_and_process_data()

    def _load_and_process_data(self):
        with open(self.fname, "r") as f:
            data = json.load(f)

        if self.data_shuffle:
            print(f"Shuffling data... {len(data)} samples")
            random.shuffle(data)

        for samples in data:
            processed = self.process_sample(samples)
            print(f"Processed sample: {processed}")  # 디버깅용 출력
            if processed:
                self.samples.append(processed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @abstractmethod
    def process_sample(
        self,
        sample: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        pass


def get_rag_context(sample: dict[str, Any], context_field: str = "retrieved_context", context_text="[관련 정보]") -> str:
    """RAG 사용 여부에 따라 context를 반환하는 함수. 예: [관련 정보] ~~~ """
    return f"{context_text} {sample.get(context_field, '')}"


def make_chat(
    inp,
    config_manager: ConfigManager,
) -> str:
    """입력 데이터를 채팅 형식으로 변환하는 함수 (버전별 프롬프트 적용)"""

    prompt_version = config_manager.system.prompt_version
    use_rag = config_manager.rag.use_rag
    context_field = config_manager.rag.context_field
    context_text = config_manager.rag.context_text

    # 버전에 맞는 instruction 가져오기
    instruction = PromptManager.get_instruction_for_type(prompt_version, inp.get('question_type', ''))

    # RAG 컨텍스트 가져오기
    if use_rag:
        context = get_rag_context(inp, context_field, context_text)
        if context:
            instruction += f" {context}"

    # 기타 정보 생성 (question과 question_type 제외)
    # other_info = {k: v for k, v in inp.items() if k not in ['question', 'question_type']}
    other_info = {k: v for k, v in inp.items() if k not in ['question']}

    # 기타 정보가 있는 경우에만 추가
    chat_parts = [instruction]
    if other_info:
        info_list = ["[기타 정보]"]
        for key, value in other_info.items():
            if value is not None and value != "":
                if config_manager.system.data_hangul_info and key in OTHER_INFO_MAP:
                    key = OTHER_INFO_MAP[key]
                info_list.append(f" {key}: {value}")
        chat_parts.append(" ".join(info_list))

    # 질문 추가
    chat_parts.append(f"[질문] {inp['question']}")

    # 최종 프롬프트 생성
    chat = " ".join(chat_parts)

    if DEBUG: print(chat)  # 디버깅용 출력

    return chat


def check_limit_length(sample, limit_length: int) -> bool:
    # 질문 길이 제한 적용
    question_text = sample.get("input", {}).get("question", "")
    question_len = len(question_text.replace(" ", ""))  # 공백 제외

    if limit_length != -1 and question_len > limit_length:
        print(f"Skipping sample due to question length: {question_len} > {limit_length}")
        return True
    return False
