import os, yaml, torch
from dataclasses import is_dataclass, fields, asdict
from typing import TypeVar, Any
from src.configs.system_config import SystemConfig
from src.configs.model_config import ModelConfig
from src.configs.sft_config import SFTConfig
from src.configs.lora_config import LoRAConfig
from src.configs.dpo_config import DPOConfig


T = TypeVar('T')


class ConfigManager:
    """싱글톤 패턴으로 구현된 설정 관리자"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.yaml_loader = YAMLLoader()
        self.type_converter = TypeConverter()

        # 로드된 설정들을 저장하는 딕셔너리
        self._configs = {}

        # 설정 파일 경로 저장
        self._config_paths = {}

        self._initialized = True

    def load_config(self, yaml_path: str, config_class: type[T], name: str = None) -> T:
        """설정을 로드하고 내부에 저장"""
        yaml_data = self.yaml_loader.load(yaml_path)
        config = self.type_converter.convert_to_dataclass(yaml_data, config_class)

        # 이름이 없으면 클래스 이름 사용
        if name is None:
            name = config_class.__name__.lower().replace('config', '')

        # 설정 저장
        self._configs[name] = config
        self._config_paths[name] = yaml_path

        return config

    def load_all_configs(self, config_dir: str = "configs"):
        """configs 디렉토리의 모든 설정 파일 로드"""


        # 설정 클래스 매핑
        config_mapping = {
            'system_config.yaml': (SystemConfig, 'system'),
            'model_config.yaml': (ModelConfig, 'model'),
            'sft_config.yaml': (SFTConfig, 'sft'),
            'lora_config.yaml': (LoRAConfig, 'lora'),
            'dpo_config.yaml': (DPOConfig, 'dpo'),
        }

        for filename, (config_class, name) in config_mapping.items():
            yaml_path = os.path.join(config_dir, filename)
            if os.path.exists(yaml_path):
                self.load_config(yaml_path, config_class, name)

    def get_config(self, name: str) -> Any:
        """저장된 설정 반환"""
        if name not in self._configs:
            raise KeyError(f"설정 '{name}'이 로드되지 않았습니다.")
        return self._configs[name]

    def get_all_configs(self) -> dict[str, Any]:
        """모든 설정 반환"""
        return self._configs.copy()

    def save_config(self, yaml_path: str, name: str = None, config_obj: Any = None) -> None:
        """설정 저장"""
        if config_obj is None and name is not None:
            # 저장된 설정 사용
            if name not in self._configs:
                raise KeyError(f"설정 '{name}'이 로드되지 않았습니다.")
            config_obj = self._configs[name]
        elif config_obj is None:
            raise ValueError("저장할 설정 객체나 이름을 지정해주세요.")

        yaml_data = self.type_converter.convert_to_dict(config_obj)
        self.yaml_loader.save(yaml_path, yaml_data)

    def save_all_configs(self, output_dir: str):
        """모든 설정을 특정 디렉토리에 저장"""
        os.makedirs(output_dir, exist_ok=True)

        for name, config in self._configs.items():
            yaml_path = os.path.join(output_dir, f"{name}_config.yaml")
            self.save_config(yaml_path, config_obj=config)

    def update_config(self, name: str, updates: dict[str, Any]) -> Any:
        """특정 설정 업데이트"""
        if name not in self._configs:
            raise KeyError(f"설정 '{name}'이 로드되지 않았습니다.")

        config = self._configs[name]
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise KeyError(f"'{key}'는 {type(config).__name__}에 존재하지 않는 속성입니다.")

        return config

    def print_all_configs(self):
        # 예쁘게 출력
        for name, config in self._configs.items():
            print(f"{name.capitalize()} Configuration:")
            print(yaml.dump(self.type_converter.convert_to_dict(config), allow_unicode=True, default_flow_style=False))
            print("-" * 40)

    def reset(self):
        """모든 설정 초기화"""
        self._configs.clear()
        self._config_paths.clear()

    @property
    def system(self) -> SystemConfig:
        """시스템 설정 빠른 접근"""
        return self.get_config('system')

    @property
    def model(self) -> ModelConfig:
        """모델 설정 빠른 접근"""
        return self.get_config('model')

    @property
    def sft(self) -> SFTConfig:
        """SFT 설정 빠른 접근"""
        return self.get_config('sft')

    @property
    def lora(self) -> LoRAConfig:
        """LoRA 설정 빠른 접근"""
        return self.get_config('lora')
    
    @property
    def dpo(self) -> DPOConfig:
        """DPO 설정 빠른 접근"""
        return self.get_config('dpo')

class YAMLLoader:
    def load(self, yaml_path: str) -> dict[str, Any]:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def save(self, yaml_path: str, data: dict[str, Any]) -> None:
        # 폴더 존재하는지 확인
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, allow_unicode=True, default_flow_style=False)


class TypeConverter:
    def __init__(self):
        # dtype 문자열 -> torch dtype 매핑
        self.dtype_map = dict(
            fp16=torch.float16,
            bf16=torch.bfloat16,
            fp32=torch.float32,
            nf4="nf4", # 특수 타입 for qlora
        )

    def convert_to_dataclass(self, yaml_data: dict[str, Any], target_class: type[T]) -> T:
        # YAML 데이터를 dataclass 인스턴스로 변환
        if 'dtype' in yaml_data and isinstance(yaml_data['dtype'], str):
            yaml_data['dtype'] = self.dtype_map.get(yaml_data['dtype'], yaml_data['dtype'])

        return target_class(**yaml_data)

    def convert_to_dict(self, obj: Any) -> dict[str, Any]:
        # dataclass instance -> dict 변환
        if not is_dataclass(obj):
            raise TypeError(f"데이터세트 클래스가 아닙니다. {type(obj)}")

        result = {}
        for field in fields(obj):
            # 클래스 속성 값 가져오기 (변수.name)
            value = getattr(obj, field.name)

            # dtype 변환
            if isinstance(value, torch.dtype):
                for dtype_str, dtype_val in self.dtype_map.items():
                    if dtype_val == value:
                        value = dtype_str
                        break

            result[field.name] = value

        return result


