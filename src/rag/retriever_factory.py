from src.rag.retriever.bm25_retriever import BM25Retriever
from src.rag.retriever.dense_retriever import DenseRetriever
from src.rag.retriever.hybrid_retriever import HybridRetriever
from src.configs.rag_config import RAGConfig

# def build_retriever(config: RAGConfig):
#     retriever_type = config.retriever_type.lower()

#     if retriever_type == "bm25":
#         return BM25Retriever(corpus_path=config.bm25_corpus_path)

#     elif retriever_type == "dense":
#         return DenseRetriever(
#             model_id=config.model_id,
#             index_path=config.dense_index_path,
#         )

#     elif retriever_type == "hybrid":
#         bm25 = BM25Retriever(corpus_path=config.bm25_corpus_path)
#         dense = DenseRetriever(
#             model_id=config.model_id,
#             index_path=config.dense_index_path,
#         )
#         return HybridRetriever(bm25=bm25, dense=dense, alpha=config.hybrid_alpha)

#     else:
#         raise ValueError(f"Unknown retriever_type: {config.retriever_type}")
