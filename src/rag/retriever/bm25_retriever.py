from rank_bm25 import BM25Okapi
from .base_retriever import BaseRetriever

class BM25Retriever(BaseRetriever):
    def __init__(self, corpus_path: str):
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.documents = [line.strip() for line in f.readlines() if line.strip()]
        self.tokenized_corpus = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int) -> list[str]:
        scores = self.bm25.get_scores(query.split())
        sorted_docs = [doc for _, doc in sorted(zip(scores, self.documents), reverse=True)]
        return sorted_docs[:top_k]
