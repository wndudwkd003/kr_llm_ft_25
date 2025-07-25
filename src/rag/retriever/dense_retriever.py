from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
from .base_retriever import BaseRetriever

class DenseRetriever(BaseRetriever):
    def __init__(self, model_id: str, index_path: str):
        self.model = SentenceTransformer(model_id)
        self.index = faiss.read_index(index_path + "/index.faiss")
        with open(index_path + "/corpus.json", "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

    def retrieve(self, query: str, top_k: int) -> list[str]:
        embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(embedding, top_k)
        return [self.corpus[i] for i in indices[0]]
