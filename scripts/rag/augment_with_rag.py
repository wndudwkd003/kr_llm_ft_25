from transformers import set_seed
from src.utils.seed_utils import set_all_seeds
from src.rag.retriever_factory import build_retriever
from src.configs.config_manager import ConfigManager
from src.utils.huggingface_utils import init_hub_env
from tqdm.auto import tqdm
from argparse import ArgumentParser
import os, json



def augment_dataset_with_rag(
        input_path,
        output_path,
        retriever,
        top_k=5,
        context_field="retrieved_context"
    ):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for example in tqdm(data, desc=f"Augmenting {os.path.basename(input_path)}"):
        question = example["input"]["question"]
        docs = retriever.retrieve(question, top_k=top_k)
        example["input"][context_field] = " ".join(docs)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main(config_manager: ConfigManager):
    # RAG 증강 실행
    rag_cfg = config_manager.rag

    retriever = build_retriever(rag_cfg)

    for split in ["train", "dev", "test"]:
        input_path = os.path.join(rag_cfg.source_dir, f"{split}.json")
        output_path = os.path.join(rag_cfg.output_dir, f"{split}.json")

        if not os.path.exists(input_path):
            print(f"[!] Skipping: {input_path} (not found)")
            continue

        augment_dataset_with_rag(
            input_path=input_path,
            output_path=output_path,
            retriever=retriever,
            top_k=rag_cfg.top_k,
            context_field=rag_cfg.context_field
        )

    print("RAG augmentation completed successfully.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run RAG augmentation on datasets")
    parser.add_argument("--config", type=str, default="configs", help="Path to the configuration directory")
    args = parser.parse_args()

    # 설정 관리자 초기화
    config_manager = ConfigManager()
    config_manager.load_all_configs(config_dir=args.config)
    config_manager.print_all_configs()

    init_hub_env(config_manager.system.hf_token)

    set_seed(config_manager.system.seed)
    set_all_seeds(config_manager.system.seed, deterministic=config_manager.system.deterministic)

    # 메인 함수 실행
    main(config_manager)
