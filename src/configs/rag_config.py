from dataclasses import field, dataclass

@dataclass
class RAGConfig:
    source_dir: str = "data/raw"
    output_dir: str = "data/raw_rag"
    chunk_size: int = 300
    chunk_overlap: int = 20
    model_id: str = "nlpai-lab/KURE-v1"
    top_k: int = 5
    retriever_type: str = "dense"
    hybrid_alpha: int|float = 0.5
    context_text: str = "[관련 정보]"
    context_field: str = "retrieved_context"

