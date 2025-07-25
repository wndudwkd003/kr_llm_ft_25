from src.data.base_dataset import BaseDataset, make_chat, check_limit_length
from src.data.prompt_manager import PromptManager

DEBUG = False

class DPODataset(BaseDataset):
    """
    TRL DPOTrainer가 요구하는 컬럼(prompt, chosen, rejected)을 제공하는 Dataset.

    Args:
        다시 작성 예정
    """
    def process_sample(self, sample):
        # 질문 길이 제한 적용
        if check_limit_length(sample, self.config_manager.system.data_question_length_limit):
            return None

        # 시스템 프롬프트 가져오기
        system_prompt = PromptManager.get_system_prompt(self.config_manager.system.prompt_version)

        # 사용자 프롬프트 생성
        user_prompt = make_chat(
            sample["input"],
            config_manager=self.config_manager,
        )

        # 메시지 구성
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 소스 토크나이즈
        prompt_input_ids = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False
        )[0] # [seq_len] -> dpo에서 이렇게 한다고 하네?

        if DEBUG: print(message)

        # 정답 / 오답 응답
        chosen_text = sample["output"].get("answer", "")
        rejected_text = sample["output"].get("rejected", "")

        if not chosen_text or not rejected_text:
            return None

        # EOS 토큰 추가
        chosen_text += self.tokenizer.eos_token
        rejected_text += self.tokenizer.eos_token

        # 각각 토크나이즈 (1D 텐서로 추출)
        chosen_input_ids = self.tokenizer(
            chosen_text,
            return_attention_mask=False,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"][0]

        rejected_input_ids = self.tokenizer(
            rejected_text,
            return_attention_mask=False,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"][0]

        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }


