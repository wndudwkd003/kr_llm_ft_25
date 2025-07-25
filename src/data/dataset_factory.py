from src.data.sft_dataset import SFTDataset
from src.data.prompt_manager import PromptVersion

class DatasetFactory:
    DATASET_TYPES = {
        "sft": SFTDataset,
        # "dpo": DPODataset,
        # "rag": RAGDataset,
    }

    @classmethod  # 이 데코레이터 추가!
    def create_dataset(
        cls,
        dataset_type: str,
        fname: str,
        tokenizer,
        prompt_version: PromptVersion = PromptVersion.V1,
        data_question_length_limit: int = 512,
        data_shuffle: bool = False,
        use_rag: bool = False,
        context_field: str = "retrieved_context"
    ):
        """데이터셋 생성"""
        dataset_class = cls.DATASET_TYPES.get(dataset_type.lower())
        if not dataset_class:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        return dataset_class(fname, tokenizer, prompt_version, data_question_length_limit, data_shuffle, use_rag, context_field)

