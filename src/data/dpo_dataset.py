from src.data.base_dataset import BaseDataset, make_chat, check_limit_length
from src.data.prompt_manager import PromptManager

DEBUG = False

class DPODataset(BaseDataset):
    """TRL DPOTrainer용: prompt / chosen / rejected 세 필드를 반환"""

    def process_sample(self, sample):
        # 0) 길이 제한 (필터링)
        if check_limit_length(sample, self.config_manager.system.data_question_length_limit):
            return None

        # 1) 시스템/유저 프롬프트 만들기
        system_prompt = PromptManager.get_system_prompt(self.config_manager.system.prompt_version)
        user_prompt   = make_chat(sample["input"], config_manager=self.config_manager)

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        print("###################################")
        print("message", message)

        # 2) prompt 텍스트 (토큰화 X, 문자열만)
        prompt_text = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            tokenize=False,
        ).strip()

        # 3) chosen / rejected 추출
        out = sample.get("output", {})
        if isinstance(out, dict):
            chosen_text   = out.get("answer", "").strip()
            rejected_text = out.get("rejected", "").strip()
        else:
            # 구조가 다르면 여기서 처리
            return None

        if not chosen_text or not rejected_text:
            return None

        eos = self.tokenizer.eos_token or ""
        chosen_text   = chosen_text   + eos
        rejected_text = rejected_text + eos

        result = {
            "prompt":   prompt_text,
            "chosen":   chosen_text,
            "rejected": rejected_text,
        }

        if DEBUG:
            print(result)

        return result
