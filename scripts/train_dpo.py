import unsloth
import os, json, argparse
from transformers import set_seed
from src.train.sft_trainer import UnslothSFTTrainer
from src.configs.config_manager import ConfigManager
from src.utils.data_utils import prepare_dataset
from src.data.dpo_dataset import SFTDataset
from src.utils.path_utils import get_output_dir

CURRENT_TRAIN_TYPE = "dpo"

def init_config_manager(config_dir: str = "configs", train_type: str = "dpo") -> ConfigManager:
    config_manager = ConfigManager()
    config_manager.load_all_configs(config_dir=config_dir)
    output_dir = get_output_dir(config_manager, train_type=train_type)
    os.makedirs(output_dir, exist_ok=True)
    config_manager.update_config("dpo", {"output_dir": output_dir, "logging_dir": os.path.join(output_dir, "logs")})
    config_manager.print_all_configs()
    return config_manager

def save_metrics(metrics, output_dir):
    """Save training metrics to a file."""
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_file}")

def main():
    # SFT 트레이너 초기화
    trainer = UnslothSFTTrainer(config_manager)

    train_dataset, eval_dataset = prepare_dataset(
        config_manager=config_manager,
        tokenizer=trainer.tokenizer,
        data_class=SFTDataset
    )

    metrics = trainer.train(train_dataset, eval_dataset)
    adapter_path = trainer.save_adapter()

    save_metrics(metrics, config_manager.dpo.output_dir)
    print(f"Training completed. Adapter saved at {adapter_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPO Model")
    parser.add_argument("--config", type=str, default="configs", help="Path to the configuration directory")
    args = parser.parse_args()

    # 설정 관리자 초기화
    config_manager = init_config_manager(config_dir=args.config, train_type=CURRENT_TRAIN_TYPE)
    config_manager.update_config("dpo", {"seed": config_manager.system.seed})
    set_seed(config_manager.system.seed)

    main()
