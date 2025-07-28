from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from ray import tune

@dataclass
class TuneConfig:
    # =============================================================================
    # Ray Tune 기본 설정
    # =============================================================================
    name: str = "llm_hyperparameter_tuning"
    num_samples: int = 50  # 총 실험 횟수
    max_concurrent_trials: int = 1  # 동시 실행할 트라이얼 수

    # 리소스 설정
    resources_per_trial: Dict[str, float] = field(default_factory=lambda: {
        "cpu": 4,
        "gpu": 1
    })

    # =============================================================================
    # 스케줄러 설정
    # =============================================================================
    scheduler_type: str = "ASHA"  # "ASHA", "PBT", "BOHB", "MedianStopping"

    # ASHA 스케줄러 설정
    asha_config: Dict[str, Any] = field(default_factory=lambda: {
        "metric": "eval_loss",
        "mode": "min",
        "max_t": 2000,  # 2000스텝으로 설정
        "grace_period": 200,  # max_t의 10% 정도
        "reduction_factor": 3,
        "brackets": 1
    })

    # =============================================================================
    # 서치 알고리즘 설정
    # =============================================================================
    search_algorithm: str = "bayesian"  # "random", "grid", "bayesian", "hyperopt"

    # =============================================================================
    # 하이퍼파라미터 서치 스페이스 정의
    # =============================================================================

    # 학습률 관련
    learning_rate: Dict[str, Any] = field(default_factory=lambda: {
        "type": "loguniform",
        "low": 1e-6,
        "high": 1e-3
    })

    warmup_ratio: Dict[str, Any] = field(default_factory=lambda: {
        "type": "uniform",
        "low": 0.01,
        "high": 0.1
    })

    weight_decay: Dict[str, Any] = field(default_factory=lambda: {
        "type": "uniform",
        "low": 0.001,
        "high": 0.1
    })

    lr_scheduler_type: Dict[str, Any] = field(default_factory=lambda: {
        "type": "choice",
        "values": ["cosine", "cosine_with_restarts", "polynomial"] # "linear"
    })

    # 배치 크기 관련
    per_device_train_batch_size: Dict[str, Any] = field(default_factory=lambda: {
        "type": "choice",
        "values": [1, 2]
    })

    gradient_accumulation_steps: Dict[str, Any] = field(default_factory=lambda: {
        "type": "choice",
        "values": [1, 2, 4, 8, 16, 32]  # GLOBAL_BATCH_SIZE // (per_device_train_batch_size * NUM_DEVICES)
    })

    # LoRA 관련 파라미터
    lora_r: Dict[str, Any] = field(default_factory=lambda: {
        "type": "choice",
        "values": [8, 16, 32, 64, 128]
    })

    lora_alpha: Dict[str, Any] = field(default_factory=lambda: {
        "type": "choice",
        "values": [8, 16, 32, 64, 128, 256]
    })

    lora_dropout: Dict[str, Any] = field(default_factory=lambda: {
        "type": "uniform",
        "low": 0.0,
        "high": 0.3
    })

    # 기존 fixed_params에서 target_modules 제거하고
    lora_target_modules: Dict[str, Any] = field(default_factory=lambda: {
        "type": "choice",
        "values": [
            ["q_proj", "v_proj"],  # 기본
            ["q_proj", "k_proj", "v_proj", "o_proj"],  # attention만
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 전체
            ["gate_proj", "up_proj", "down_proj"]  # MLP만
        ]
    })

    # =============================================================================
    # 고정 파라미터 (튜닝하지 않는 파라미터들)
    # =============================================================================
    fixed_params: Dict[str, Any] = field(default_factory=lambda: {
        "seed": 2025,
        "num_train_epochs": 5,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "logging_steps": 25,
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "push_to_hub": False,
        "optim": "adamw_torch",
        "bias": "none",
        "init_lora_weights": True,
        "label_names": ["labels"]
    })

    # =============================================================================
    # 출력 및 로깅 설정
    # =============================================================================
    output_dir: str = "./outputs/tune_results"
    local_dir: str = "./ray_results"

    # 체크포인트 설정
    checkpoint_config: Dict[str, Any] = field(default_factory=lambda: {
        "checkpoint_score_attr": "eval_loss",
        "checkpoint_freq": 1,
        "num_to_keep": 3
    })

    # =============================================================================
    # 조기 중단 설정
    # =============================================================================
    early_stopping: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "patience": 3,
        "min_delta": 0.001
    })

    # =============================================================================
    # 리포팅 설정
    # =============================================================================
    progress_reporter: Dict[str, Any] = field(default_factory=lambda: {
        "metric_columns": ["eval_loss", "eval_accuracy", "training_iteration"],
        "max_progress_rows": 20,
        "max_error_rows": 5,
        "max_column_length": 20
    })

    # =============================================================================
    # 베스트 모델 선택 기준
    # =============================================================================
    best_model_selection: Dict[str, Any] = field(default_factory=lambda: {
        "metric": "eval_loss",
        "mode": "min",
        "scope": "all"  # "all", "last", "avg"
    })

    def to_search_space(self) -> Dict[str, Any]:
        """Ray Tune 서치 스페이스로 변환"""
        search_space = {}

        # 각 하이퍼파라미터를 Ray Tune 서치 스페이스로 변환
        for param_name in [
            "learning_rate", "warmup_ratio", "weight_decay", "lr_scheduler_type",
            "per_device_train_batch_size", "gradient_accumulation_steps",
            "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules"
        ]:
            param_config = getattr(self, param_name)
            search_space[param_name] = self._convert_to_tune_space(param_config)

        return search_space

    def _convert_to_tune_space(self, param_config: Dict[str, Any]):
        """파라미터 설정을 Ray Tune 서치 스페이스로 변환"""
        param_type = param_config["type"]

        if param_type == "loguniform":
            return tune.loguniform(param_config["low"], param_config["high"])
        elif param_type == "uniform":
            return tune.uniform(param_config["low"], param_config["high"])
        elif param_type == "choice":
            return tune.choice(param_config["values"])
        elif param_type == "randint":
            return tune.randint(param_config["low"], param_config["high"])
        elif param_type == "randn":
            return tune.randn(param_config["mean"], param_config["sd"])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    def get_scheduler(self):
        """설정된 스케줄러 반환"""
        if self.scheduler_type == "ASHA":
            from ray.tune.schedulers import ASHAScheduler
            return ASHAScheduler(**self.asha_config)
        elif self.scheduler_type == "PBT":
            from ray.tune.schedulers import PopulationBasedTraining
            # PBT 설정 추가 필요
            pass
        # 다른 스케줄러들도 필요시 추가

    def get_search_algorithm(self):
        """설정된 서치 알고리즘 반환"""
        if self.search_algorithm == "bayesian":
            from ray.tune.search.bayesopt import BayesOptSearch
            return BayesOptSearch()
        elif self.search_algorithm == "hyperopt":
            from ray.tune.search.hyperopt import HyperOptSearch
            return HyperOptSearch()
        # 기본적으로는 random search 사용
        return None
