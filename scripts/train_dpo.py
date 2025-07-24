# 모델 실행 코드 작성

import os, argparse
from src.configs.config_manager import ConfigManager

def main(args):
    config_manager = ConfigManager()
    config_manager.load_all_configs(config_dir=args.config)

    print(config_manager.get_all_configs())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPO Model")
    parser.add_argument("--config", type=str, default="configs", help="Path to the configuration directory")
    args = parser.parse_args()


    main(args)
