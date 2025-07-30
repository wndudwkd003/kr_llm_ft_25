import unsloth
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import set_seed
from src.utils.seed_utils import set_all_seeds
from src.utils.huggingface_utils import init_hub_env
from src.configs.config_manager import ConfigManager
from src.data.prompt_manager import PromptManager
from src.data.base_dataset import make_chat
from tqdm.auto import tqdm
from datetime import datetime
import os, json, argparse, hashlib, torch

CURRENT_TEST_TYPE = "dpo"


# ─────────────────────────────────────────────────────
# 0) 설정 로드 util (변경 없음)
# ─────────────────────────────────────────────────────
def init_config_manager_for_test(save_dir: str = "configs") -> ConfigManager:
    cm = ConfigManager()
    cm.load_all_configs(config_dir=os.path.join(save_dir, "configs"))

    adapter_dir      = os.path.join(save_dir, "lora_adapter")   # DPO adapter
    test_result_dir  = os.path.join(save_dir, "test_result")
    os.makedirs(test_result_dir, exist_ok=True)

    cm.update_config("system", {
        "save_dir": save_dir,
        "adapter_dir": adapter_dir,
        "test_result_dir": test_result_dir,
    })
    cm.print_all_configs()
    return cm


# ─────────────────────────────────────────────────────
# 1) 모델 + 어댑터 로드 🔥
# ─────────────────────────────────────────────────────
def get_merged_dpo_model(cm):
    # ── 0) 베이스 모델 로드 ────────────────────────────────
    print("▶ Loading base model …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = cm.model.model_id,
        max_seq_length = cm.model.max_seq_length,
        dtype          = cm.model.dtype,
        load_in_4bit   = False,
        load_in_8bit   = False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 1) SFT LoRA 병합 ──────────────────────────────────
    sft_adapter_dir = os.path.join(cm.system.sft_model_for_dpo, "lora_adapter")
    if not os.path.isdir(sft_adapter_dir):
        raise FileNotFoundError(f"SFT adapter not found: {sft_adapter_dir}")
    print(f"▶ Merging SFT adapter: {sft_adapter_dir}")

    peft_sft = PeftModel.from_pretrained(model, sft_adapter_dir, is_trainable=False)
    model    = peft_sft.merge_and_unload()        # LoRA 가중치를 베이스에 합침
    del peft_sft
    torch.cuda.empty_cache()

    # ── 2) DPO LoRA 적용 ─────────────────────────────────
    dpo_adapter_dir = os.path.join(cm.dpo.output_dir, "lora_adapter")
    if not os.path.isdir(dpo_adapter_dir):
        raise FileNotFoundError(f"DPO adapter not found: {dpo_adapter_dir}")
    print(f"▶ Loading DPO adapter: {dpo_adapter_dir}")

    model = PeftModel.from_pretrained(model, dpo_adapter_dir, is_trainable=False)

    # ── 3) Inference 최적화 ──────────────────────────────
    model = FastLanguageModel.for_inference(model)
    return model, tokenizer


# ─────────────────────────────────────────────────────
# 2) 실제 테스트
# ─────────────────────────────────────────────────────
def main(cm: ConfigManager):
    model, tokenizer = get_merged_dpo_model(cm)
    model.eval()

    with open(os.path.join(cm.system.data_raw_dir, "test.json"), "r", encoding="utf-8") as f:
        test_data = json.load(f)

    system_prompt = PromptManager.get_system_prompt(cm.system.prompt_version)
    results = []

    for sample in tqdm(test_data, desc="Testing", unit="sample"):
        user_prompt = make_chat(sample["input"], cm)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        attention_mask = (inputs != tokenizer.pad_token_id).long().to(model.device)

        outputs = model.generate(
            inputs,
            max_new_tokens   = cm.model.max_new_tokens,
            do_sample        = cm.model.do_sample,
            attention_mask   = attention_mask,
        )

        answer = tokenizer.decode(
            outputs[0][inputs.shape[-1]:],
            skip_special_tokens=True,
        ).lstrip("답변: ").split("#")[0].strip()

        results.append({
            "id": sample["id"],
            "input": sample["input"],
            "output": {"answer": answer},
        })

    save_hash  = hashlib.md5(cm.system.save_dir.encode()).hexdigest()[:8]
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_fname  = f"test_results_{save_hash}_{timestamp}.json"
    out_path   = os.path.join(cm.system.test_result_dir, out_fname)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved to: {out_path}")


# ─────────────────────────────────────────────────────
# 3) run
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test merged DPO model")
    parser.add_argument("--save_dir", required=True, help="디렉터리(root) 경로 – DPO 학습 결과가 들어있는 폴더")
    args = parser.parse_args()

    cm = init_config_manager_for_test(args.save_dir)
    cm.update_config("dpo", {"seed": cm.system.seed})

    init_hub_env(cm.system.hf_token)
    set_seed(cm.system.seed)
    set_all_seeds(cm.system.seed, deterministic=cm.system.deterministic)

    main(cm)
