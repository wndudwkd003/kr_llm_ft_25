import torch
from src.data.prompt_manager import PromptManager
from src.data.base_dataset import BaseDataset, make_chat, check_limit_length

DEBUG = False

class SFTDataset(BaseDataset):
    def process_sample(self, sample):
        # 0) 질문 길이 제한
        if check_limit_length(sample, self.config_manager.system.data_question_length_limit):
            return None

        # 1) 시스템/유저 프롬프트
        system_prompt = PromptManager.get_system_prompt(self.config_manager.system.prompt_version)
        user_prompt = make_chat(sample["input"], config_manager=self.config_manager)

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        if DEBUG: print(message)

        # 2) 소스 토크나이즈
        source = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False
        )

        # dict / tensor 모두 대응
        if isinstance(source, dict):
            source_ids = source["input_ids"][0]
        else:
            source_ids = source[0]

        # 3) 타겟 텍스트 준비
        raw_output = sample.get("output", "")
        if isinstance(raw_output, dict):
            target_text = raw_output.get("answer", "")
        else:
            target_text = raw_output

        if not target_text:
            return None

        target_text = target_text + self.tokenizer.eos_token  # EOS

        target = self.tokenizer(
            target_text,
            return_attention_mask=False,
            add_special_tokens=False,
            return_tensors="pt",
        )
        target_ids = target["input_ids"][0].to(torch.long)

        # 4) 길이 맞추기 (★중요★)
        max_len = getattr(self.config_manager.model, "max_seq_length", 4096)

        # (필요시) 너무 긴 source 자체 컷
        if source_ids.size(0) >= max_len:
            # 방법 A: 샘플 버림
            # return None
            # 방법 B: source 뒤쪽을 잘라냄
            source_ids = source_ids[-(max_len - 1):]  # EOS 자리 등 고려
            # 이후 target_ids는 거의 못 들어가므로 room 계산에서 걸러짐

        room_for_target = max_len - source_ids.size(0)
        if room_for_target <= 0:
            return None  # target 한 토큰도 못 들어가면 버리기(선택)

        if target_ids.size(0) > room_for_target:
            target_ids = target_ids[:room_for_target]

        # 5) 최종 concat
        input_ids = torch.cat((source_ids, target_ids), dim=0)

        ignore_prefix = torch.full(
            (source_ids.size(0),),
            self.IGNORE_INDEX,
            dtype=torch.long,
        )
        labels = torch.cat((ignore_prefix, target_ids), dim=0)

        # 6) sanity check
        if input_ids.size(0) != labels.size(0):
            # 혹시라도 불일치시 바로 드롭
            return None

        if DEBUG:
            print("len(input_ids) =", input_ids.size(0))
            print("len(labels)    =", labels.size(0))

        return {"input_ids": input_ids, "labels": labels}
