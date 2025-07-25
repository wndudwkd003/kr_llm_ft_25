import numpy as np
from src.rag.embedder import TextEmbedder
from src.rag.vector_store import load_faiss_index
from src.rag.retriever.base_retriever import BaseRetriever
from src.configs.rag_config import RAGConfig


class DenseRetriever(BaseRetriever):
    def __init__(
        self,
        rag_cfg: RAGConfig,
    ):
        self.batch_size = rag_cfg.batch_size
        self.embedder = TextEmbedder(model_name=rag_cfg.model_id)

        self.index, self.corpus = load_faiss_index(
            index_dir=rag_cfg.index_dir,
            index_name=rag_cfg.index_name,
            corpus_name=rag_cfg.corpus_name,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> list[str]:
        query_embedding = self.embedder.encode([query], self.batch_size)[0]

        # D = distances, I = indices
        D, I = self.index.search(np.array([query_embedding]), top_k)
        return [self.corpus[i] for i in I[0]]

    def retrieve_batch(self, queries: list[str], top_k: int = 5) -> list[list[str]]:
        query_embeddings = self.embedder.encode(queries, self.batch_size)  # (B, D)
        D, I = self.index.search(np.array(query_embeddings), top_k)
        return [[self.corpus[i] for i in row] for row in I]
